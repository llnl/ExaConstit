/*
 * test_mask_seeding: seeding and morphology restricted to the active elements of a mesh by an
 * element mask (mesh/element_mask.hpp).
 *
 * Setup. A box of 4 x 4 x 6 cells of size 0.25 (z from 0 to 1.5) whose bottom two layers
 * (z < 0.5) are inactive, like a build plate under a part. The reference is a separate mesh of
 * only the active part: 4 x 4 x 4 cells shifted up by 0.5. All coordinates are multiples of 0.25,
 * so the two meshes have bitwise the same element corners (and, for the curved variant, the same
 * Nodes values, since the warp is a function of the position only); two inactive layers keep the
 * parity of the cell index, so MFEM splits the active cells into the same tets in both meshes.
 *
 * What is checked, and why:
 *  - Seeds (rejection, element-weighted, lattice, Johnson-Mehl, ROI filler) placed on the masked
 *    mesh over MPI_COMM_WORLD equal, bit for bit, the seeds placed on the reference mesh on one
 *    rank: an inactive element must behave exactly as if it were not there (no candidates accepted
 *    in it, no volume, no box, no element-weighted stream).
 *  - Surface clearance: the interface z = 0.5 between active and inactive elements is bonded, not
 *    a surface, so seeds may lie close to it, while they keep the clearance from the true outer
 *    surface.
 *  - Weighted samples and Lloyd relaxation: the masked samples cover the active volume only, and
 *    the relaxed seeds equal those of the reference mesh.
 *  - nullptr / empty mask: unchanged results.
 *  - A curved plate (order-2 Nodes, only plate nodes moved) under straight active elements: the
 *    straightness test looks at the active elements only, so the seeds equal those of the
 *    straight reference.
 *  - NearestVertex with a mask: a rank that holds a copy of the winning vertex only through
 *    inactive elements (every plate element on the last rank) still reports its local copy.
 * Every run uses np = 1-4 (the masked mesh is distributed over MPI_COMM_WORLD).
 */
#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/mesh/element_mask.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/mesh/vertex_coordinates.hpp"
#include "microstructure/morphology/lloyd.hpp"
#include "microstructure/seeding/johnson_mehl.hpp"
#include "microstructure/seeding/material_point_locator.hpp"
#include "microstructure/seeding/roi_filler.hpp"
#include "microstructure/seeding/seed_element_locator.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

constexpr int ACTIVE_ATTRIBUTE = 1; ///< attribute of the active part
constexpr int PLATE_ATTRIBUTE = 2;  ///< attribute of the inactive plate
constexpr double PLATE_TOP = 0.5;   ///< height of the plate's top face

/** Smooth warp of the interior that fixes the box faces' planes (a function of x only). */
void Warp(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    y(0) += 0.03 * std::sin(4.0 * x(0)) * std::sin(3.0 * x(1));
    y(1) += 0.02 * std::sin(3.0 * x(0) + x(2));
}

/** Masked box (plate attribute below z = 0.5) or the active-only reference, optionally curved. */
mfem::Mesh BuildMesh(bool reference, mfem::Element::Type type, int curved_order) {
    mfem::Mesh mesh = reference ? mfem::Mesh::MakeCartesian3D(4, 4, 4, type, 1.0, 1.0, 1.0, false)
                                : mfem::Mesh::MakeCartesian3D(4, 4, 6, type, 1.0, 1.0, 1.5, false);
    if (reference) {
        mesh.Transform([](const mfem::Vector& x, mfem::Vector& y) {
            y = x;
            y(2) += PLATE_TOP;
        });
    } else {
        for (int e = 0; e < mesh.GetNE(); e++) {
            mfem::Vector center(3);
            mfem::ElementTransformation* trans = mesh.GetElementTransformation(e);
            trans->Transform(mfem::Geometries.GetCenter(mesh.GetElementBaseGeometry(e)), center);
            mesh.SetAttribute(e, center(2) < PLATE_TOP ? PLATE_ATTRIBUTE : ACTIVE_ATTRIBUTE);
        }
        mesh.SetAttributes();
    }
    if (curved_order > 0) {
        mesh.SetCurvature(curved_order);
        mesh.Transform(Warp);
    }
    return mesh;
}

/** A serial mesh distributed over comm, with the mask of its active attribute. */
struct MaskedMesh {
    ms::SerialOrder order;                ///< partitioning and local-to-serial element map
    std::unique_ptr<mfem::ParMesh> pmesh; ///< the distributed mesh
    ms::ElementMask mask;                 ///< active elements (attribute ACTIVE_ATTRIBUTE)
};

/** Distributes `serial` over `comm` and builds the mask. Collective. */
MaskedMesh Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    MaskedMesh m;
    m.order = ms::BuildSerialOrder(serial, comm);
    m.pmesh = ms::MakeParMesh(comm, serial, m.order);
    m.mask.active.SetSize(m.pmesh->GetNE());
    for (int e = 0; e < m.pmesh->GetNE(); e++) {
        m.mask.active[e] = (m.pmesh->GetAttribute(e) == ACTIVE_ATTRIBUTE) ? 1 : 0;
    }
    return m;
}

/** Bitwise equality of positions and source ids. */
void ExpectSameSeeds(const ms::SeedSet& a, const ms::SeedSet& b, const std::string& what) {
    ASSERT_EQ(a.Size(), b.Size()) << what;
    std::size_t differ = 0;
    for (std::size_t i = 0; i < a.Size(); i++) {
        const bool same = a.x[i] == b.x[i] && a.y[i] == b.y[i] && a.z[i] == b.z[i] &&
                          a.source_id[i] == b.source_id[i];
        differ += same ? 0 : 1;
    }
    EXPECT_EQ(differ, 0u) << what;
}

/** One test mesh: element type and curved order (0: straight). */
struct Variant {
    mfem::Element::Type type; ///< hex or tet
    int curved_order;         ///< order of the curved Nodes, 0 for none
    const char* name;         ///< name in the test output
};

/** The test meshes: straight hexes and tets, and an order-2 curved hex. */
const Variant kVariants[] = {{mfem::Element::HEXAHEDRON, 0, "hex"},
                             {mfem::Element::TETRAHEDRON, 0, "tet"},
                             {mfem::Element::HEXAHEDRON, 2, "curved hex p=2"}};

} // namespace

// Every seed generator gives, on the masked mesh, exactly the seeds of the active-only mesh: the
// mesh box, the material volume, the candidates accepted and (element-weighted) the per-element
// streams depend only on the active elements. The masked mesh runs on all ranks with a shared
// locator built once (as the driver does); the reference runs on one rank without a mask.
TEST(MaskSeeding, SeedsEqualActiveOnlyMesh) {
    for (const Variant& v : kVariants) {
        mfem::Mesh masked_serial = BuildMesh(false, v.type, v.curved_order);
        mfem::Mesh reference_serial = BuildMesh(true, v.type, v.curved_order);
        MaskedMesh masked = Distribute(masked_serial, MPI_COMM_WORLD);
        MaskedMesh reference = Distribute(reference_serial, MPI_COMM_SELF);
        const ms::MaterialPointLocator locator(
            *masked.pmesh, ms::IsStraightSided(*masked.pmesh, MPI_COMM_WORLD), &masked.mask);
        ms::MaterialContext context;
        context.mask = &masked.mask;
        context.locator = &locator;

        const std::pair<ms::SeedPlacement, const char*> placements[] = {
            {ms::SeedPlacement::REJECTION, "rejection"},
            {ms::SeedPlacement::ELEMENT_WEIGHTED, "element-weighted"},
            {ms::SeedPlacement::LATTICE, "lattice"}};
        for (const auto& placement : placements) {
            ms::SeedingParameters params;
            params.placement = placement.first;
            params.num_seeds = 40;
            params.rng_seed = 11;
            params.lattice_jitter = 0.2;
            ms::SeedingReport report_m;
            ms::SeedingReport report_r;
            const ms::SeedSet a = ms::GenerateSeeds(*masked.pmesh, params, &report_m, &context);
            const ms::SeedSet b = ms::GenerateSeeds(*reference.pmesh, params, &report_r);
            ExpectSameSeeds(a, b, std::string(v.name) + " " + placement.second);
            EXPECT_EQ(report_m.material_volume, report_r.material_volume) << v.name;
            EXPECT_EQ(report_m.bbox_lo, report_r.bbox_lo) << v.name;
            EXPECT_EQ(report_m.bbox_hi, report_r.bbox_hi) << v.name;
            // No seed below the plate top (inactive elements never receive seeds)
            for (const double z : a.z) {
                EXPECT_GE(z, PLATE_TOP) << v.name << " " << placement.second;
            }
        }

        // Johnson-Mehl nucleation
        ms::JohnsonMehlParameters jm;
        jm.nucleation_rate = 200.0;
        jm.growth_speed = 1.0;
        jm.num_nuclei = 25;
        jm.rng_seed = 5;
        const ms::SeedSet jm_a = ms::GenerateJohnsonMehlSeeds(*masked.pmesh, jm, nullptr, &context);
        const ms::SeedSet jm_b = ms::GenerateJohnsonMehlSeeds(*reference.pmesh, jm);
        ExpectSameSeeds(jm_a, jm_b, std::string(v.name) + " Johnson-Mehl");

        // ROI filler: filler seeds outside a box inside the active part
        ms::RoiFillerParameters roi;
        roi.lo = {0.25, 0.25, 0.75};
        roi.hi = {0.75, 0.75, 1.25};
        roi.num_filler = 20;
        roi.rng_seed = 3;
        roi.source_id_offset = 1000;
        const ms::RoiFillerResult roi_a = ms::FillOutsideRoi(
            *masked.pmesh, MPI_COMM_WORLD, roi, 10, &context);
        const ms::RoiFillerResult roi_b = ms::FillOutsideRoi(
            *reference.pmesh, MPI_COMM_SELF, roi, 10);
        ExpectSameSeeds(roi_a.filler, roi_b.filler, std::string(v.name) + " ROI filler");
        EXPECT_EQ(roi_a.roi_material_volume, roi_b.roi_material_volume) << v.name;
        EXPECT_EQ(roi_a.outside_material_volume, roi_b.outside_material_volume) << v.name;
    }
}

// The surface clearance measures the distance to the true outer surface of the active elements
// only. The interface z = 0.5 with the inactive plate is bonded, so seeds may come closer to it
// than the clearance (expected: about 0.1 / 1 of the 200 seeds lie within 0.1 above it), while
// every seed keeps the clearance from the side faces x, y = 0, 1 and the top z = 1.5. On the
// active-only reference mesh, whose bottom z = 0.5 is a real surface, no seed comes that close.
TEST(MaskSeeding, ClearanceIgnoresTheInterface) {
    const double c = 0.1;
    for (const Variant& v : kVariants) {
        mfem::Mesh masked_serial = BuildMesh(false, v.type, 0);
        mfem::Mesh reference_serial = BuildMesh(true, v.type, 0);
        MaskedMesh masked = Distribute(masked_serial, MPI_COMM_WORLD);
        MaskedMesh reference = Distribute(reference_serial, MPI_COMM_SELF);
        ms::MaterialContext context;
        context.mask = &masked.mask;
        ms::SeedingParameters params;
        params.num_seeds = 200;
        params.rng_seed = 19;
        params.boundary_clearance = c;
        const ms::SeedSet a = ms::GenerateSeeds(*masked.pmesh, params, nullptr, &context);
        const ms::SeedSet b = ms::GenerateSeeds(*reference.pmesh, params);
        int near_interface_a = 0;
        int near_interface_b = 0;
        for (std::size_t i = 0; i < a.Size(); i++) {
            const double side = std::min(
                {a.x[i], 1.0 - a.x[i], a.y[i], 1.0 - a.y[i], 1.5 - a.z[i]});
            EXPECT_GE(side, c) << v.name;
            near_interface_a += (a.z[i] - PLATE_TOP < c) ? 1 : 0;
        }
        for (std::size_t i = 0; i < b.Size(); i++) {
            near_interface_b += (b.z[i] - PLATE_TOP < c) ? 1 : 0;
        }
        EXPECT_GT(near_interface_a, 0) << v.name;
        EXPECT_EQ(near_interface_b, 0) << v.name;
    }
}

// Weighted samples and Lloyd relaxation. Element and quadrature samples of the masked mesh cover
// the active elements only: their global count and total weight equal those of the reference
// mesh. Lloyd relaxation (3 steps, clamped to the material through the shared masked locator)
// then moves the seeds exactly as on the reference mesh: the cell measures are fixed-point sums
// over the same samples, whose order does not matter.
TEST(MaskSeeding, SamplesAndLloydEqualActiveOnlyMesh) {
    for (const Variant& v : kVariants) {
        mfem::Mesh masked_serial = BuildMesh(false, v.type, v.curved_order);
        mfem::Mesh reference_serial = BuildMesh(true, v.type, v.curved_order);
        MaskedMesh masked = Distribute(masked_serial, MPI_COMM_WORLD);
        MaskedMesh reference = Distribute(reference_serial, MPI_COMM_SELF);
        const ms::MaterialPointLocator locator(
            *masked.pmesh, ms::IsStraightSided(*masked.pmesh, MPI_COMM_WORLD), &masked.mask);
        ms::MaterialContext context;
        context.mask = &masked.mask;
        context.locator = &locator;

        for (const bool quadrature : {false, true}) {
            ms::WeightedSamples sa;
            ms::WeightedSamples sb;
            if (quadrature) {
                ms::MakeQuadratureWeightedSamples(*masked.pmesh, -1, sa, &masked.mask);
                ms::MakeQuadratureWeightedSamples(*reference.pmesh, -1, sb);
            } else {
                ms::MakeElementWeightedSamples(*masked.pmesh, sa, &masked.mask);
                ms::MakeElementWeightedSamples(*reference.pmesh, sb);
            }
            long long counts[2] = {sa.Size(), sb.Size()};
            MPI_Allreduce(MPI_IN_PLACE, &counts[0], 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            EXPECT_EQ(counts[0], counts[1]) << v.name;

            ms::SeedingParameters params;
            params.num_seeds = 12;
            params.rng_seed = 23;
            const ms::SeedSet seeds = ms::GenerateSeeds(*reference.pmesh, params);
            const ms::CellMeasureAccumulator acc_a(MPI_COMM_WORLD, sa, 12);
            const ms::CellMeasureAccumulator acc_b(MPI_COMM_SELF, sb, 12);
            mfem::Vector one_a(sa.Size());
            mfem::Vector one_b(sb.Size());
            one_a = 1.0;
            one_b = 1.0;
            EXPECT_EQ(acc_a.Integrate(one_a), acc_b.Integrate(one_b)) << v.name; // volume

            ms::LloydParameters lloyd;
            lloyd.iterations = 3;
            const ms::SeedSet ra = ms::RunLloyd(
                *masked.pmesh, MPI_COMM_WORLD, sa, seeds, lloyd, {}, nullptr, &context);
            const ms::SeedSet rb = ms::RunLloyd(
                *reference.pmesh, MPI_COMM_SELF, sb, seeds, lloyd, {});
            ExpectSameSeeds(ra, rb, std::string(v.name) + (quadrature ? " Lloyd qp" : " Lloyd"));
        }
    }
}

// A null context and a context with an empty mask (every element active) give the same seeds as
// before masks existed (the unmasked path is unchanged; the stage 1a-1f tests check that against
// stored results).
TEST(MaskSeeding, EmptyMaskChangesNothing) {
    mfem::Mesh serial = BuildMesh(false, mfem::Element::TETRAHEDRON, 0);
    MaskedMesh m = Distribute(serial, MPI_COMM_WORLD);
    ms::ElementMask empty;
    ms::MaterialContext context;
    context.mask = &empty;
    for (const auto placement :
         {ms::SeedPlacement::REJECTION, ms::SeedPlacement::ELEMENT_WEIGHTED}) {
        ms::SeedingParameters params;
        params.placement = placement;
        params.num_seeds = 30;
        params.rng_seed = 2;
        params.boundary_clearance = 0.05;
        ExpectSameSeeds(ms::GenerateSeeds(*m.pmesh, params),
                        ms::GenerateSeeds(*m.pmesh, params, nullptr, &context),
                        "empty mask");
    }
}

// Only the plate is curved (order-2 Nodes moved below z = 0.5, with zero displacement and slope
// at the interface): the active elements are straight, so seeding uses the straight element
// boxes and draws exactly the seeds of the straight active-only reference with order-2 Nodes,
// whether the locator is shared (built as the driver does) or built inside GenerateSeeds.
TEST(MaskSeeding, CurvedPlateDoesNotCurveTheMaterial) {
    const auto warp_plate = [](const mfem::Vector& x, mfem::Vector& y) {
        y = x;
        const double depth = std::max(0.0, PLATE_TOP - x(2));
        y(0) += 0.3 * depth * depth * std::sin(5.0 * x(1));
        y(1) += 0.2 * depth * depth * std::cos(4.0 * x(0));
    };
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        const std::string name = type == mfem::Element::HEXAHEDRON ? "hex" : "tet";
        mfem::Mesh masked_serial = BuildMesh(false, type, 0);
        masked_serial.SetCurvature(2);
        masked_serial.Transform(warp_plate);
        mfem::Mesh reference_serial = BuildMesh(true, type, 0);
        reference_serial.SetCurvature(2);
        MaskedMesh masked = Distribute(masked_serial, MPI_COMM_WORLD);
        MaskedMesh reference = Distribute(reference_serial, MPI_COMM_SELF);
        ASSERT_FALSE(ms::IsStraightSided(*masked.pmesh, MPI_COMM_WORLD)) << name;
        const bool straight = ms::IsStraightSided(*masked.pmesh, MPI_COMM_WORLD, &masked.mask);
        EXPECT_TRUE(straight) << name;
        const ms::MaterialPointLocator locator(*masked.pmesh, straight, &masked.mask);
        ms::MaterialContext shared;
        shared.mask = &masked.mask;
        shared.locator = &locator;
        ms::MaterialContext local;
        local.mask = &masked.mask;
        for (const auto placement :
             {ms::SeedPlacement::REJECTION, ms::SeedPlacement::ELEMENT_WEIGHTED}) {
            ms::SeedingParameters params;
            params.placement = placement;
            params.num_seeds = 40;
            params.rng_seed = 17;
            const ms::SeedSet b = ms::GenerateSeeds(*reference.pmesh, params);
            ExpectSameSeeds(ms::GenerateSeeds(*masked.pmesh, params, nullptr, &shared),
                            b,
                            name + " shared locator");
            ExpectSameSeeds(ms::GenerateSeeds(*masked.pmesh, params, nullptr, &local),
                            b,
                            name + " own locator");
        }
    }
}

// NearestVertex with a mask on a partition that puts every plate element on the last rank (at
// np >= 2 that rank has no active element: its mask is all zero). The seeds below the plate top are
// nearest to vertices on the interface z = 0.5, which the last rank holds only through plate
// elements. Every rank holding a copy of the winning vertex must report it in local_vertex (the
// documented contract), and the winners are vertices of active elements.
TEST(MaskSeeding, NearestVertexReportsEveryCopy) {
    mfem::Mesh serial = BuildMesh(false, mfem::Element::HEXAHEDRON, 0);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MaskedMesh m;
    const int ne = serial.GetNE();
    m.order.partitioning.SetSize(ne);
    int num_active = 0;
    for (int e = 0; e < ne; e++) {
        num_active += serial.GetAttribute(e) == ACTIVE_ATTRIBUTE ? 1 : 0;
    }
    std::vector<int> local;
    for (int e = 0, k = 0; e < ne; e++) {
        int owner = 0;
        if (nranks > 1) {
            owner = serial.GetAttribute(e) == ACTIVE_ATTRIBUTE ? (k++ * (nranks - 1)) / num_active
                                                               : nranks - 1;
        }
        m.order.partitioning[e] = owner;
        if (owner == rank) {
            local.push_back(e);
        }
    }
    m.order.local_to_serial.SetSize(static_cast<int>(local.size()));
    for (std::size_t i = 0; i < local.size(); i++) {
        m.order.local_to_serial[static_cast<int>(i)] = local[i];
    }
    m.pmesh = ms::MakeParMesh(MPI_COMM_WORLD, serial, m.order);
    const ms::ElementMask* mask = nullptr;
    if (m.pmesh->GetNE() > 0) {
        m.mask.active.SetSize(m.pmesh->GetNE());
        for (int e = 0; e < m.pmesh->GetNE(); e++) {
            m.mask.active[e] = (m.pmesh->GetAttribute(e) == ACTIVE_ATTRIBUTE) ? 1 : 0;
        }
        mask = &m.mask; // all zero on the last rank at np >= 2
    }

    ms::SeedSet seeds;
    seeds.x = {0.5, 0.25, 0.6};
    seeds.y = {0.5, 0.75, 0.6};
    seeds.z = {0.45, 0.3, 1.0};
    seeds.source_id = {0, 1, 2};
    const std::vector<int> ids = {0, 1, 2};
    const ms::NearestVertexResult result = ms::NearestVertex(
        *m.pmesh, MPI_COMM_WORLD, seeds, ids, mask);
    const ms::VertexCoordinates coords(*m.pmesh);
    for (std::size_t k = 0; k < ids.size(); k++) {
        const auto& p = result.position[k];
        EXPECT_GE(p[2], PLATE_TOP) << "seed " << k; // a vertex of an active element
        int copy = -1;
        for (int v = 0; v < coords.NumVertices(); v++) {
            const double* x = coords.Vertex(v);
            if (x[0] == p[0] && x[1] == p[1] && x[2] == p[2]) {
                copy = v;
            }
        }
        EXPECT_EQ(result.local_vertex[k], copy) << "seed " << k << ", rank " << rank;
    }
    EXPECT_EQ(result.position[0][2], PLATE_TOP);
    if (nranks > 1 && rank == nranks - 1) {
        EXPECT_GE(result.local_vertex[0], 0); // held only through plate elements
    }
}

/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
