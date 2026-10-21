#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/preflight.hpp"
#include "microstructure/mesh/serial_order.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

/*
 * Tests for the mesh module: serial-order bookkeeping for distributed meshes, the element and
 * quadrature-point samplers, and the preflight checks. Each test runs at np = 1..4 (see
 * test_mesh.cmake), so every MPI-dependent result is checked for several partitionings. All
 * tests are called by every rank (the functions under test are collective or the test reduces
 * results with MPI_Allreduce).
 */

namespace ms = exaconstit::microstructure;

namespace {

/** @brief This process's rank in `comm`. */
int Rank(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

/** @brief Number of ranks in MPI_COMM_WORLD. */
int NumRanks() {
    int nranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    return nranks;
}

/** @brief Sum of `v` over all ranks of MPI_COMM_WORLD (collective). */
double GlobalSum(double v) {
    double out = 0.0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

/** @brief Structured mesh of [0,1]^3 with nx*ny*nz cells (each split into 6 tets for TET). */
mfem::Mesh UnitCube(int nx, int ny, int nz, mfem::Element::Type type) {
    return mfem::Mesh::MakeCartesian3D(nx, ny, nz, type, 1.0, 1.0, 1.0, false);
}

/** @brief Average of the element's vertices: the volume centroid of an affine hex or a tet. */
void VertexAverage(const mfem::Mesh& mesh, int e, double c[3]) {
    mfem::Array<int> verts;
    mesh.GetElementVertices(e, verts);
    c[0] = 0.0;
    c[1] = 0.0;
    c[2] = 0.0;
    for (int v : verts) {
        const double* x = mesh.GetVertex(v);
        for (int d = 0; d < 3; d++) {
            c[d] += x[d] / verts.Size();
        }
    }
}

/** @brief A ParMesh together with the serial order it was built with. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** @brief Partitions `serial` over MPI_COMM_WORLD and builds the ParMesh (collective). */
Distributed Distribute(mfem::Mesh& serial) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    d.pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, d.order);
    return d;
}

/**
 * @brief Host pointers to the sampler outputs.
 *
 * The samplers write their mfem::Vector outputs wherever the mfem::Device runs kernels (GPU
 * memory in a GPU run). HostRead() returns a read-only host pointer, copying the data back
 * from the GPU first if needed, so the checks below work for every Device configuration.
 */
struct HostElementSamples {
    explicit HostElementSamples(const ms::ElementSamples& s)
        : size(s.volume.Size()), cx(s.cx.HostRead()), cy(s.cy.HostRead()), cz(s.cz.HostRead()),
          volume(s.volume.HostRead()) {}
    int size;
    const mfem::real_t* cx;
    const mfem::real_t* cy;
    const mfem::real_t* cz;
    const mfem::real_t* volume;
};

/** @brief Host pointers to QuadraturePointSamples outputs (see HostElementSamples). */
struct HostQuadraturePointSamples {
    explicit HostQuadraturePointSamples(const ms::QuadraturePointSamples& s)
        : size(s.x.Size()), x(s.x.HostRead()), y(s.y.HostRead()), z(s.z.HostRead()),
          weight(s.weight.HostRead()) {}
    int size;
    const mfem::real_t* x;
    const mfem::real_t* y;
    const mfem::real_t* z;
    const mfem::real_t* weight;
};

} // namespace

/*
 * local_to_serial must be strictly increasing (ParMesh keeps serial order), cover every serial
 * element exactly once over all ranks, and point at the right element: local element i and
 * serial element local_to_serial[i] must have the same center. Centers are compared with
 * EXPECT_EQ (exact equality): ParMesh copies the vertex coordinates bit for bit and both
 * meshes compute the center with the same arithmetic.
 */
TEST(MicrostructureMesh, SerialOrderMatchesSerialMesh) {
    mfem::Mesh serial = UnitCube(4, 3, 5, mfem::Element::HEXAHEDRON);
    Distributed d = Distribute(serial);
    const ms::SerialOrder& order = d.order;

    // EXPECT, not ASSERT: returning early on one rank would hang the others' collectives
    EXPECT_EQ(order.partitioning.Size(), serial.GetNE());
    for (int i = 1; i < order.local_to_serial.Size(); i++) {
        EXPECT_LT(order.local_to_serial[i - 1], order.local_to_serial[i]);
    }
    int total = 0;
    int local = order.local_to_serial.Size();
    MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(total, serial.GetNE());

    // Local element i is serial element local_to_serial[i]
    mfem::Vector c_local(3);
    mfem::Vector c_serial(3);
    for (int i = 0; i < d.pmesh->GetNE(); i++) {
        d.pmesh->GetElementCenter(i, c_local);
        serial.GetElementCenter(order.local_to_serial[i], c_serial);
        for (int k = 0; k < 3; k++) {
            EXPECT_EQ(c_local(k), c_serial(k));
        }
    }
}

/*
 * Element attributes are copied from the serial mesh to the ParMesh, so gathering each local
 * element's attribute back to rank 0 in serial order must reproduce the serial attributes
 * exactly. The attribute pattern (e*7)%5+1 is deliberately non-monotone so that a wrong
 * ordering cannot pass by accident. A wrong-sized input must throw on every rank.
 */
TEST(MicrostructureMesh, GatherToSerialReproducesAttributes) {
    for (auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = UnitCube(3, 3, 3, type);
        for (int e = 0; e < serial.GetNE(); e++) {
            serial.SetAttribute(e, (e * 7) % 5 + 1);
        }
        serial.SetAttributes();
        Distributed d = Distribute(serial);

        mfem::Array<int> local(d.pmesh->GetNE());
        for (int i = 0; i < local.Size(); i++) {
            local[i] = d.pmesh->GetAttribute(i);
        }
        const int root = 0;
        const std::vector<int> gathered = ms::GatherToSerial(MPI_COMM_WORLD, d.order, local, root);
        if (Rank() == root) {
            // EXPECT, not ASSERT: returning early on one rank would hang the others' collectives
            EXPECT_EQ(static_cast<int>(gathered.size()), serial.GetNE());
            if (static_cast<int>(gathered.size()) == serial.GetNE()) {
                for (int e = 0; e < serial.GetNE(); e++) {
                    EXPECT_EQ(gathered[static_cast<std::size_t>(e)], serial.GetAttribute(e));
                }
            }
        } else {
            EXPECT_TRUE(gathered.empty());
        }

        mfem::Array<int> wrong_size(local.Size() + 1);
        EXPECT_THROW(ms::GatherToSerial(MPI_COMM_WORLD, d.order, wrong_size, root),
                     std::runtime_error);
    }
}

/*
 * The consistency checks in MakeParMesh and GatherToSerial are reduced over all ranks: when
 * the input is wrong on one rank only (here the last rank), every rank must throw, otherwise
 * the healthy ranks would wait forever in the next collective call. Reaching the end of the
 * test at all on every rank proves no rank was left behind.
 */
TEST(MicrostructureMesh, ConsistencyChecksThrowOnAllRanks) {
    mfem::Mesh serial = UnitCube(2, 2, 3, mfem::Element::HEXAHEDRON);
    const bool is_last_rank = Rank() == NumRanks() - 1;

    // Partitioning with the wrong length on one rank: caught before the ParMesh is built
    ms::SerialOrder bad_partitioning = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    if (is_last_rank) {
        bad_partitioning.partitioning.Append(0);
    }
    EXPECT_THROW(ms::MakeParMesh(MPI_COMM_WORLD, serial, bad_partitioning), std::runtime_error);

    // local_to_serial with one entry too many on one rank: caught after the ParMesh is built
    ms::SerialOrder bad_local = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    if (is_last_rank) {
        bad_local.local_to_serial.Append(serial.GetNE());
    }
    EXPECT_THROW(ms::MakeParMesh(MPI_COMM_WORLD, serial, bad_local), std::runtime_error);

    // Wrong number of local values on one rank only
    Distributed d = Distribute(serial);
    mfem::Array<int> values(d.pmesh->GetNE() + (is_last_rank ? 1 : 0));
    values = 0;
    EXPECT_THROW(ms::GatherToSerial(MPI_COMM_WORLD, d.order, values, 0), std::runtime_error);
}

/*
 * Linear (affine) hexes and tets: the true volume centroid equals the vertex average, and the
 * element volumes sum to the cube volume 1. For the 2x3x4 hex grid every element has volume
 * 1/(2*3*4) = 1/24. Sampling must not attach Nodes to a vertex-only mesh.
 */
TEST(MicrostructureMesh, SamplerLinearHexAndTet) {
    for (auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = UnitCube(2, 3, 4, type);
        Distributed d = Distribute(serial);

        ms::ElementSamples samples;
        ms::ComputeElementSamples(*d.pmesh, -1, samples);
        const HostElementSamples h(samples);
        // EXPECT, not ASSERT: returning early on one rank would hang the others' collectives
        EXPECT_EQ(h.size, d.pmesh->GetNE());
        const int num_checked = (h.size == d.pmesh->GetNE()) ? d.pmesh->GetNE() : 0;

        double local_volume = 0.0;
        for (int i = 0; i < num_checked; i++) {
            local_volume += h.volume[i];
            double expected[3];
            VertexAverage(serial, d.order.local_to_serial[i], expected);
            EXPECT_NEAR(h.cx[i], expected[0], 1e-14);
            EXPECT_NEAR(h.cy[i], expected[1], 1e-14);
            EXPECT_NEAR(h.cz[i], expected[2], 1e-14);
            if (type == mfem::Element::HEXAHEDRON) {
                EXPECT_NEAR(h.volume[i], 1.0 / 24.0, 1e-15);
            }
        }
        EXPECT_NEAR(GlobalSum(local_volume), 1.0, 1e-13);
        // Sampling must not give a vertex-only mesh Nodes
        EXPECT_EQ(d.pmesh->GetNodes(), nullptr);
    }
}

/*
 * Curved mesh with a known answer. The map x' = x (1 + a y), y' = y, z' = z of the unit cube
 * is quadratic, so order-2 Nodes represent it exactly, and detJ = 1 + a y. Integrating over
 * the unit cube:
 *   volume = int (1 + a y) = 1 + a/2
 *   xbar   = int x (1 + a y)^2 / volume = (1/2)(1 + a + a^2/3) / volume
 *   ybar   = int y (1 + a y) / volume = (1/2 + a/3) / volume
 *   zbar   = 1/2 by symmetry.
 * The default quadrature order for p = 2 is 4*2 - 1 = 7, which integrates these exactly, so
 * the volume-weighted sum of the element centroids must match to round-off.
 */
TEST(MicrostructureMesh, SamplerCurvedMeshAnalyticVolume) {
    const double a = 0.6;
    const double volume = 1.0 + a / 2.0;
    const double xbar = 0.5 * (1.0 + a + a * a / 3.0) / volume;
    const double ybar = (0.5 + a / 3.0) / volume;

    for (auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = UnitCube(3, 3, 3, type);
        serial.SetCurvature(2);
        serial.Transform([a](const mfem::Vector& x, mfem::Vector& y) {
            y = x;
            y(0) = x(0) * (1.0 + a * x(1));
        });
        Distributed d = Distribute(serial);
        // EXPECT, not ASSERT: returning early on one rank would hang the others' collectives
        EXPECT_NE(d.pmesh->GetNodes(), nullptr);
        EXPECT_EQ(ms::DefaultSampleQuadOrder(*d.pmesh), 7);

        ms::ElementSamples samples;
        ms::ComputeElementSamples(*d.pmesh, -1, samples);
        const HostElementSamples h(samples);
        double v = 0.0;
        double mx = 0.0;
        double my = 0.0;
        double mz = 0.0;
        for (int i = 0; i < d.pmesh->GetNE(); i++) {
            v += h.volume[i];
            mx += h.volume[i] * h.cx[i];
            my += h.volume[i] * h.cy[i];
            mz += h.volume[i] * h.cz[i];
        }
        const double gv = GlobalSum(v);
        EXPECT_NEAR(gv, volume, 1e-13);
        EXPECT_NEAR(GlobalSum(mx) / gv, xbar, 1e-13);
        EXPECT_NEAR(GlobalSum(my) / gv, ybar, 1e-13);
        EXPECT_NEAR(GlobalSum(mz) / gv, 0.5, 1e-13);
    }
}

/*
 * Ranks that own no elements (more ranks than elements: a 2-element curved mesh at np = 3, 4)
 * must still be able to call the samplers and preflight. The samplers are not collective, so
 * an empty rank returns empty arrays immediately; preflight is collective and every rank,
 * empty or not, joins the same reductions. The total volume of the curved map
 * x' = x (1 + a y) on [0,1]x[0,1]x[0,2] is 2 (1 + a/2).
 */
TEST(MicrostructureMesh, EmptyRanksSampleAndPreflight) {
    const double a = 0.4;
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(
        1, 1, 2, mfem::Element::HEXAHEDRON, 1.0, 1.0, 2.0, false);
    serial.SetCurvature(2);
    serial.Transform([a](const mfem::Vector& x, mfem::Vector& y) {
        y = x;
        y(0) = x(0) * (1.0 + a * x(1));
    });
    Distributed d = Distribute(serial);

    ms::ElementSamples samples;
    ms::ComputeElementSamples(*d.pmesh, -1, samples);
    const HostElementSamples h(samples);
    // EXPECT, not ASSERT: returning early on one rank would hang the others' collectives
    EXPECT_EQ(h.size, d.pmesh->GetNE());
    const int num_checked = (h.size == d.pmesh->GetNE()) ? d.pmesh->GetNE() : 0;
    double v = 0.0;
    for (int i = 0; i < num_checked; i++) {
        v += h.volume[i];
    }
    EXPECT_NEAR(GlobalSum(v), 2.0 * (1.0 + a / 2.0), 1e-13);

    ms::QuadraturePointSamples qp;
    ms::ComputeQuadraturePointSamples(*d.pmesh, -1, 0, d.pmesh->GetNE(), qp);
    const HostQuadraturePointSamples hq(qp);
    if (d.pmesh->GetNE() == 0) {
        EXPECT_EQ(qp.num_qp, 0);
        EXPECT_EQ(hq.size, 0);
    }

    const ms::PreflightReport report = ms::CheckMesh(*d.pmesh);
    EXPECT_TRUE(report.Passed()) << ms::DescribePreflightFailures(report);
    EXPECT_EQ(report.num_elements, 2);
}

/*
 * The quadrature-point sampler and the element sampler use the same rule, so for each element
 * of the requested range the weights must sum to the element volume and the weighted mean of
 * the point coordinates must equal the element centroid. A 2x2x2 Gauss rule (8 points)
 * integrates order 3 on hexes. An out-of-range request must throw.
 */
TEST(MicrostructureMesh, QuadraturePointSamples) {
    mfem::Mesh serial = UnitCube(3, 2, 2, mfem::Element::HEXAHEDRON);
    Distributed d = Distribute(serial);
    const int ne = d.pmesh->GetNE();

    ms::ElementSamples samples;
    ms::ComputeElementSamples(*d.pmesh, 3, samples);
    const HostElementSamples h(samples);

    // Second half of the local elements
    const int begin = ne / 2;
    ms::QuadraturePointSamples qp;
    ms::ComputeQuadraturePointSamples(*d.pmesh, 3, begin, ne, qp);
    const HostQuadraturePointSamples hq(qp);
    EXPECT_EQ(qp.elem_begin, begin);
    EXPECT_EQ(qp.elem_end, ne);
    if (ne > 0) {
        EXPECT_EQ(qp.num_qp, 8); // 2x2x2 Gauss points integrate order 3
    }
    // EXPECT, not ASSERT: returning early on one rank would hang the others' collectives
    const int expected_size = qp.num_qp * (ne - begin);
    EXPECT_EQ(hq.size, expected_size);
    const int check_end = (hq.size == expected_size) ? ne : begin;

    for (int e = begin; e < check_end; e++) {
        double wsum = 0.0;
        double mx = 0.0;
        double my = 0.0;
        double mz = 0.0;
        for (int q = 0; q < qp.num_qp; q++) {
            const int k = (e - begin) * qp.num_qp + q;
            wsum += hq.weight[k];
            mx += hq.weight[k] * hq.x[k];
            my += hq.weight[k] * hq.y[k];
            mz += hq.weight[k] * hq.z[k];
        }
        EXPECT_NEAR(wsum, h.volume[e], 1e-15);
        EXPECT_NEAR(mx / wsum, h.cx[e], 1e-14);
        EXPECT_NEAR(my / wsum, h.cy[e], 1e-14);
        EXPECT_NEAR(mz / wsum, h.cz[e], 1e-14);
    }

    EXPECT_THROW(ms::ComputeQuadraturePointSamples(*d.pmesh, 3, 0, ne + 1, qp),
                 std::runtime_error);
}

/*
 * Structured hex and tet meshes of the unit cube satisfy every requirement: the report passes,
 * counts all elements, has a positive minimum detJ and produces no failure text.
 */
TEST(MicrostructureMesh, PreflightPassesValidMeshes) {
    for (auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = UnitCube(3, 3, 3, type);
        Distributed d = Distribute(serial);
        const ms::PreflightReport report = ms::CheckMesh(*d.pmesh);
        EXPECT_TRUE(report.Passed()) << ms::DescribePreflightFailures(report);
        EXPECT_EQ(report.num_elements, serial.GetNE());
        EXPECT_GT(report.min_detj, 0.0);
        EXPECT_TRUE(ms::DescribePreflightFailures(report).empty());
        EXPECT_NO_THROW(ms::ThrowOnPreflightFailure(report));
    }
}

/*
 * A folded element. In the 2x2x2 hex grid the center vertex (0.5,0.5,0.5) is shared by all 8
 * elements; for element 7 (the cell [0.5,1]^3 in lexicographic order) it is the corner
 * opposite (1,1,1). The three neighbors of that corner along the cell edges, (1,0.5,0.5),
 * (0.5,1,0.5) and (0.5,0.5,1), span the plane x+y+z = 2. Moving the center vertex to
 * (0.95,0.95,0.95), where x+y+z = 2.85, carries it across that plane, which turns the corner
 * inside out: detJ is negative near that corner of element 7. Preflight must flag it, and
 * mapping the flagged local ids back to serial ids must include serial element 7 and agree
 * with the global count.
 */
TEST(MicrostructureMesh, PreflightFlagsInvertedElement) {
    mfem::Mesh serial = UnitCube(2, 2, 2, mfem::Element::HEXAHEDRON);
    for (int v = 0; v < serial.GetNV(); v++) {
        double* x = serial.GetVertex(v);
        if (std::abs(x[0] - 0.5) + std::abs(x[1] - 0.5) + std::abs(x[2] - 0.5) < 1e-12) {
            x[0] = 0.95;
            x[1] = 0.95;
            x[2] = 0.95;
        }
    }
    Distributed d = Distribute(serial);
    const ms::PreflightReport report = ms::CheckMesh(*d.pmesh);
    EXPECT_FALSE(report.Passed());
    EXPECT_TRUE(report.single_geometry);
    EXPECT_TRUE(report.conforming);
    EXPECT_GE(report.num_bad_elements, 1);
    EXPECT_LT(report.min_detj, 0.0);
    EXPECT_THROW(ms::ThrowOnPreflightFailure(report), std::runtime_error);

    // Mark flagged local elements with 1 and gather the marks in serial order
    mfem::Array<int> bad(d.pmesh->GetNE());
    bad = 0;
    for (int e : report.local_bad_elements) {
        bad[e] = 1;
    }
    const std::vector<int> serial_bad = ms::GatherToSerial(MPI_COMM_WORLD, d.order, bad, 0);
    if (Rank() == 0) {
        EXPECT_EQ(serial_bad[7], 1);
        int count = 0;
        for (int b : serial_bad) {
            count += b;
        }
        EXPECT_EQ(count, report.num_bad_elements);
    }
}

/*
 * Mirroring the mesh (x -> -x) reverses the orientation of every element, so detJ < 0
 * everywhere: every element must be flagged, globally and on each rank.
 */
TEST(MicrostructureMesh, PreflightFlagsMirroredMesh) {
    mfem::Mesh serial = UnitCube(2, 2, 3, mfem::Element::TETRAHEDRON);
    for (int v = 0; v < serial.GetNV(); v++) {
        serial.GetVertex(v)[0] *= -1.0;
    }
    Distributed d = Distribute(serial);
    const ms::PreflightReport report = ms::CheckMesh(*d.pmesh);
    EXPECT_EQ(report.num_bad_elements, serial.GetNE());
    EXPECT_EQ(static_cast<int>(report.local_bad_elements.size()), d.pmesh->GetNE());
}

/*
 * Mixed element geometries must be detected both when they are mixed within one rank's mesh
 * (a hex plus a disjoint tet, checked on MPI_COMM_SELF; the message names "Cube") and when
 * each rank is uniform but ranks differ (hexes on even ranks, tets on odd ranks). The second
 * case passes only at np = 1, where there are no odd ranks.
 */
TEST(MicrostructureMesh, PreflightFlagsMixedMesh) {
    // Local mixed mesh: a unit hex and a disjoint tet above it. Mesh(dim, nvert, nelem, nbdr,
    // sdim) reserves space for 12 vertices and 2 elements.
    mfem::Mesh mixed(3, 12, 2, 0, 3);
    const double hex_vertices[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    for (const auto& x : hex_vertices) {
        mixed.AddVertex(x[0], x[1], x[2]);
    }
    mixed.AddVertex(0.0, 0.0, 2.0);
    mixed.AddVertex(1.0, 0.0, 2.0);
    mixed.AddVertex(0.0, 1.0, 2.0);
    mixed.AddVertex(0.0, 0.0, 3.0);
    mixed.AddHex(0, 1, 2, 3, 4, 5, 6, 7);
    mixed.AddTet(8, 9, 10, 11);
    mixed.FinalizeMesh();

    const ms::PreflightReport local_report = ms::CheckMesh(mixed, MPI_COMM_SELF);
    EXPECT_FALSE(local_report.single_geometry);
    EXPECT_EQ(local_report.num_bad_elements, 0);
    EXPECT_NE(ms::DescribePreflightFailures(local_report).find("Cube"), std::string::npos);

    // Globally mixed but locally uniform: hexes on even ranks, tets on odd ranks
    const int nranks = NumRanks();
    mfem::Mesh local = UnitCube(
        1, 1, 1, (Rank() % 2 == 0) ? mfem::Element::HEXAHEDRON : mfem::Element::TETRAHEDRON);
    const ms::PreflightReport global_report = ms::CheckMesh(local, MPI_COMM_WORLD);
    EXPECT_EQ(global_report.single_geometry, nranks == 1);
    EXPECT_EQ(global_report.Passed(), nranks == 1);
}

/*
 * A mesh converted to MFEM's nonconforming representation (EnsureNCMesh) must fail the
 * conforming check, and MakeParMesh must refuse it because its local element order is not tied
 * to the serial order. A fully periodic cube (Mesh::MakePeriodic) has L2 Nodes and must fail
 * the continuous-nodes check while passing the others. The sampler itself still works on the
 * periodic mesh: the element volumes sum to the cube volume 1.
 */
TEST(MicrostructureMesh, PreflightFlagsNonconformingAndPeriodic) {
    mfem::Mesh nc = UnitCube(2, 2, 2, mfem::Element::HEXAHEDRON);
    nc.EnsureNCMesh();
    const ms::PreflightReport nc_report = ms::CheckMesh(nc, MPI_COMM_SELF);
    EXPECT_FALSE(nc_report.conforming);
    EXPECT_FALSE(nc_report.Passed());
    ms::SerialOrder order = ms::BuildSerialOrder(nc, MPI_COMM_WORLD);
    EXPECT_THROW(ms::MakeParMesh(MPI_COMM_WORLD, nc, order), std::runtime_error);

    // Periodic in x, y and z: vertices on opposite faces are identified
    mfem::Mesh base = UnitCube(3, 3, 3, mfem::Element::HEXAHEDRON);
    std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0, 0.0}),
                                              mfem::Vector({0.0, 1.0, 0.0}),
                                              mfem::Vector({0.0, 0.0, 1.0})};
    mfem::Mesh periodic =
        mfem::Mesh::MakePeriodic(base, base.CreatePeriodicVertexMapping(translations));
    const ms::PreflightReport p_report = ms::CheckMesh(periodic, MPI_COMM_SELF);
    EXPECT_FALSE(p_report.continuous_nodes);
    EXPECT_TRUE(p_report.conforming);
    EXPECT_EQ(p_report.num_bad_elements, 0);
    EXPECT_FALSE(p_report.Passed());

    // The sampler still works on the periodic (L2 Nodes) mesh
    ms::ElementSamples samples;
    ms::ComputeElementSamples(periodic, -1, samples);
    const HostElementSamples h(samples);
    double v = 0.0;
    for (int i = 0; i < periodic.GetNE(); i++) {
        v += h.volume[i];
    }
    EXPECT_NEAR(v, 1.0, 1e-13);
}

// mfem::Mpi::Init initializes MPI and finalizes it automatically at exit, as in the other
// microstructure tests. Every rank runs all tests, since most of them make collective calls.
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
