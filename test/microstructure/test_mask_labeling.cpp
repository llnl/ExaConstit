/*
 * test_mask_labeling: labeling, connected and geodesic distance modes and fragment cleanup
 * restricted to the active elements of a mesh by an element mask (mesh/element_mask.hpp).
 *
 * Setup. A grid of 4 x 2 x 6 cells of size 0.25 (x in [0, 1], z in [0, 1.5]). The bottom two
 * layers (z < 0.5) are an inactive plate. Above it the active part is a U turned upside down: two
 * columns (x < 0.25 and x > 0.75) joined by a bridge in the top layer (z > 1.25); the cells in
 * between are void (not in the mesh). The reference mesh holds only the active cells, with the
 * same vertex coordinates and the same tet split, so elements are matched by element key.
 *
 * Seeds: s0 at the bottom of the left column (0.125, 0.25, 0.6) and s1 at the top of the left
 * column (0.125, 0.25, 1.4). The bottom of the right column is nearest to s0 in a straight line
 * (0.75 versus 1.08) and through the plate (about 1.0), but through the active material (up the
 * left column, over the bridge, down the right column) it is nearer to s1 (about 1.5 versus
 * 2.3). So connected and geodesic labels there must be s1: if the plate were used as material,
 * they would be s0.
 *
 * What is checked: Euclidean (centroid and vote), connected, geodesic (FIM and graph) labels and
 * fragment cleanup on the masked mesh (MPI_COMM_WORLD, np = 1-4) equal those on the reference
 * mesh (one rank), element by element; inactive elements carry the placeholders (seed -1,
 * grain 0); and an empty mask changes nothing. Curved variants (order-2 Nodes): with the active
 * part warped, the masked geodesic modes use the nodal field and still equal the reference; with
 * only the plate warped, the active elements are straight, so the masked run takes the corner
 * field exactly like the straight reference (the straightness test looks at active elements).
 */
#include "microstructure/labeling/connected_labeler.hpp"
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/labeling/geodesic_fim.hpp"
#include "microstructure/labeling/geodesic_graph.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/labeling/vertex_field.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/element_mask.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/mesh/straight_sided.hpp"
#include "microstructure/seeding/material_point_locator.hpp"

#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

constexpr int ACTIVE_ATTRIBUTE = 1; ///< attribute of the active (generated) cells
constexpr int PLATE_ATTRIBUTE = 2;  ///< attribute of the inactive plate
constexpr int NX = 4;               ///< cells along x
constexpr int NY = 2;               ///< cells along y
constexpr int NZ = 6;               ///< cells along z
constexpr double H = 0.25;          ///< cell edge length

/** True for the plate cells: the two bottom layers (z < 0.5). */
bool IsPlate(int, int, int k) {
    return k < 2;
}

/** True for the active cells: the two columns and the top bridge above the plate. */
bool IsActive(int i, int, int k) {
    return k >= 2 && (i == 0 || i == NX - 1 || k == NZ - 1);
}

/**
 * Grid of NX x NY x NZ cells of size H keeping the plate and active cells (masked mesh) or the
 * active cells only (reference); hexes, or 6 Kuhn tets per cell (the same split in both meshes,
 * since it depends only on the cell). Attributes: plate 2, active 1.
 */
mfem::Mesh BuildMesh(bool reference, bool tets) {
    auto kept = [&](int i, int j, int k) {
        return IsActive(i, j, k) || (!reference && IsPlate(i, j, k));
    };
    auto vid = [](int i, int j, int k) {
        return i + (NX + 1) * (j + (NY + 1) * k);
    };
    int num_cells = 0;
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                num_cells += kept(i, j, k) ? 1 : 0;
            }
        }
    }
    mfem::Mesh mesh(3, (NX + 1) * (NY + 1) * (NZ + 1), tets ? 6 * num_cells : num_cells, 0, 3);
    for (int k = 0; k <= NZ; k++) {
        for (int j = 0; j <= NY; j++) {
            for (int i = 0; i <= NX; i++) {
                mesh.AddVertex(i * H, j * H, k * H);
            }
        }
    }
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if (!kept(i, j, k)) {
                    continue;
                }
                const int attribute = IsPlate(i, j, k) ? PLATE_ATTRIBUTE : ACTIVE_ATTRIBUTE;
                int c[8];
                for (int b = 0; b < 8; b++) {
                    c[b] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                }
                if (!tets) {
                    const int hex[8] = {c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]};
                    mesh.AddHex(hex, attribute);
                } else {
                    const int orders[6][3] = {
                        {1, 2, 4}, {1, 4, 2}, {2, 1, 4}, {2, 4, 1}, {4, 1, 2}, {4, 2, 1}};
                    for (const auto& o : orders) {
                        const int tet[4] = {c[0], c[o[0]], c[o[0] + o[1]], c[7]};
                        mesh.AddTet(tet, attribute);
                    }
                }
            }
        }
    }
    if (tets) {
        mesh.FinalizeTetMesh(1, 1, true);
    } else {
        mesh.FinalizeHexMesh(1, 1, true);
    }
    mesh.RemoveUnusedVertices();
    return mesh;
}

/** The distributed mesh, its mask (active attribute) and adjacency. */
struct Run {
    ms::SerialOrder order;                        ///< partitioning and local-to-serial element map
    std::unique_ptr<mfem::ParMesh> pmesh;         ///< the distributed mesh
    ms::ElementMask mask;                         ///< active elements (attribute ACTIVE_ATTRIBUTE)
    std::unique_ptr<ms::MeshAdjacency> adjacency; ///< face adjacency of the whole mesh
    MPI_Comm comm = MPI_COMM_NULL;                ///< communicator of the run
};

/** Distributes `serial` over `comm` and builds the mask and the adjacency. Collective. */
std::unique_ptr<Run> Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    auto run = std::make_unique<Run>();
    run->comm = comm;
    run->order = ms::BuildSerialOrder(serial, comm);
    run->pmesh = ms::MakeParMesh(comm, serial, run->order);
    run->mask.active.SetSize(run->pmesh->GetNE());
    for (int e = 0; e < run->pmesh->GetNE(); e++) {
        run->mask.active[e] = (run->pmesh->GetAttribute(e) == ACTIVE_ATTRIBUTE) ? 1 : 0;
    }
    run->adjacency = std::make_unique<ms::MeshAdjacency>(*run->pmesh);
    return run;
}

/** Labels of the active elements keyed by element key, over all ranks (collective). */
std::map<std::uint64_t, int>
ByKey(const Run& run, const mfem::Array<int>& labels, const ms::ElementMask* mask) {
    const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(*run.pmesh);
    std::vector<std::uint64_t> pairs;
    const int* l = labels.HostRead();
    for (int e = 0; e < run.pmesh->GetNE(); e++) {
        if (ms::ElementIsActive(mask, e)) {
            pairs.push_back(keys[static_cast<std::size_t>(e)]);
            pairs.push_back(static_cast<std::uint64_t>(static_cast<std::int64_t>(l[e])));
        }
    }
    int nranks = 1;
    MPI_Comm_size(run.comm, &nranks);
    const int count = static_cast<int>(pairs.size());
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, run.comm);
    std::vector<int> displs(static_cast<std::size_t>(nranks), 0);
    int total = 0;
    for (int r = 0; r < nranks; r++) {
        displs[static_cast<std::size_t>(r)] = total;
        total += counts[static_cast<std::size_t>(r)];
    }
    std::vector<std::uint64_t> all(static_cast<std::size_t>(total));
    MPI_Allgatherv(pairs.data(),
                   count,
                   MPI_UINT64_T,
                   all.data(),
                   counts.data(),
                   displs.data(),
                   MPI_UINT64_T,
                   run.comm);
    std::map<std::uint64_t, int> out;
    for (std::size_t i = 0; i + 1 < all.size(); i += 2) {
        out[all[i]] = static_cast<int>(static_cast<std::int64_t>(all[i + 1]));
    }
    return out;
}

/** Every inactive local element carries `placeholder`, on every rank (collective). */
bool InactiveCarry(const Run& run, const mfem::Array<int>& labels, int placeholder) {
    int ok = 1;
    const int* l = labels.HostRead();
    for (int e = 0; e < run.pmesh->GetNE(); e++) {
        if (!run.mask.IsActive(e) && l[e] != placeholder) {
            ok = 0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, run.comm);
    return ok == 1;
}

/** Seeds s0 at the bottom and s1 at the top of the left column (see the file comment). */
ms::SeedSet TwoSeeds() {
    ms::SeedSet seeds;
    seeds.x = {0.125, 0.125};
    seeds.y = {0.25, 0.25};
    seeds.z = {0.6, 1.4};
    seeds.source_id = {0, 1};
    return seeds;
}

/** Key of the element whose centroid is nearest to (x, y, z), over all ranks (collective). */
std::uint64_t KeyNear(const Run& run, double x, double y, double z) {
    std::pair<double, std::uint64_t> best{1e300, 0};
    if (run.pmesh->GetNE() > 0) {
        ms::ElementSamples samples;
        ms::ComputeElementSamples(*run.pmesh, -1, samples);
        const std::vector<std::uint64_t> keys = ms::ComputeElementKeys(*run.pmesh);
        for (int e = 0; e < run.pmesh->GetNE(); e++) {
            const double d2 = std::pow(samples.cx(e) - x, 2) + std::pow(samples.cy(e) - y, 2) +
                              std::pow(samples.cz(e) - z, 2);
            best = std::min(best, std::make_pair(d2, keys[static_cast<std::size_t>(e)]));
        }
    }
    struct {
        double d2;
        int rank;
    } in{best.first, 0}, out{};
    MPI_Comm_rank(run.comm, &in.rank);
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, run.comm);
    std::uint64_t key = best.second;
    MPI_Bcast(&key, 1, MPI_UINT64_T, out.rank, run.comm);
    return key;
}

/** Smooth warp of every node (genuinely curved elements). */
void WarpAll(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    y(0) += 0.02 * std::sin(5.0 * x(2)) * std::cos(3.0 * x(1));
    y(1) += 0.015 * std::sin(4.0 * x(0) + x(2));
}

/** Warp of the plate only: zero, with zero slope, at and above the plate top z = 2H. */
void WarpPlate(const mfem::Vector& x, mfem::Vector& y) {
    y = x;
    const double depth = std::max(0.0, 2.0 * H - x(2));
    y(0) += 0.3 * depth * depth * std::sin(5.0 * x(1));
    y(1) += 0.2 * depth * depth * std::cos(4.0 * x(0));
}

} // namespace

// Euclidean labels (centroid and quadrature vote) and fragment cleanup. The masked run equals the
// reference run element by element; inactive elements carry seed -1 and grain 0. The cleanup
// input is the Euclidean label of s0 / s1, which leaves s0's grain in two pieces in the active
// part (the left column bottom and the right column bottom, joined only through the plate): with
// the plate as a free surface both runs find the same fragments and give the same result.
TEST(MaskLabeling, EuclideanLabelsAndCleanupEqualActiveOnlyMesh) {
    for (const bool tets : {false, true}) {
        mfem::Mesh masked_serial = BuildMesh(false, tets);
        mfem::Mesh reference_serial = BuildMesh(true, tets);
        auto masked = Distribute(masked_serial, MPI_COMM_WORLD);
        auto reference = Distribute(reference_serial, MPI_COMM_SELF);
        const ms::SeedSet seeds = TwoSeeds();
        for (const bool vote : {false, true}) {
            ms::TessellationOptions tess;
            if (vote) {
                tess.sampling = ms::SamplingMode::QUADRATURE_VOTE;
                tess.vote_levels = 1;
            }
            const ms::TessellationLabels a = ms::LabelTessellation(
                *masked->pmesh, MPI_COMM_WORLD, seeds, tess, &masked->mask);
            const ms::TessellationLabels b = ms::LabelTessellation(
                *reference->pmesh, MPI_COMM_SELF, seeds, tess);
            EXPECT_EQ(ByKey(*masked, a.element_seed, &masked->mask),
                      ByKey(*reference, b.element_seed, nullptr));
            EXPECT_EQ(a.num_grains, b.num_grains);
            EXPECT_EQ(a.num_split_elements > 0, b.num_split_elements > 0);
            EXPECT_TRUE(InactiveCarry(*masked, a.element_seed, -1));
            EXPECT_TRUE(InactiveCarry(*masked, a.element_grain, 0));

            ms::FragmentCleanupParameters cleanup;
            cleanup.policy = ms::FragmentPolicy::SPLIT;
            const ms::FragmentCleanupResult ca = ms::CleanupFragments(
                *masked->adjacency, a.element_grain, a.num_grains, {}, cleanup, &masked->mask);
            const ms::FragmentCleanupResult cb = ms::CleanupFragments(
                *reference->adjacency, b.element_grain, b.num_grains, {}, cleanup);
            EXPECT_EQ(ByKey(*masked, ca.element_grain, &masked->mask),
                      ByKey(*reference, cb.element_grain, nullptr));
            EXPECT_EQ(ca.num_grains, cb.num_grains);
            EXPECT_EQ(ca.report.fragments_found, cb.report.fragments_found);
            EXPECT_GT(cb.report.fragments_found, 0); // s0's grain really is in two pieces
            EXPECT_TRUE(InactiveCarry(*masked, ca.element_grain, 0));
        }
    }
}

// Connected and geodesic (FIM and graph) modes: paths run through active elements only. The
// masked run equals the reference run element by element, and the bottom of the right column
// belongs to s1, reached over the bridge, not to s0, which is nearer in a straight line and
// through the plate. The shared masked locator is used as the driver does.
TEST(MaskLabeling, ConnectedAndGeodesicPathsAvoidThePlate) {
    for (const bool tets : {false, true}) {
        mfem::Mesh masked_serial = BuildMesh(false, tets);
        mfem::Mesh reference_serial = BuildMesh(true, tets);
        auto masked = Distribute(masked_serial, MPI_COMM_WORLD);
        auto reference = Distribute(reference_serial, MPI_COMM_SELF);
        const ms::MaterialPointLocator locator(
            *masked->pmesh, ms::IsStraightSided(*masked->pmesh, MPI_COMM_WORLD), &masked->mask);
        ms::MaterialContext context;
        context.mask = &masked->mask;
        context.locator = &locator;
        const ms::SeedSet seeds = TwoSeeds();
        const std::uint64_t right_bottom = KeyNear(*reference, 0.875, 0.25, 0.625);

        // Connected: Euclidean labels constrained to the material
        ms::TessellationOptions tess;
        mfem::Array<int> sa;
        mfem::Array<int> sb;
        ms::LabelElementsWithSeeds(*masked->pmesh, seeds, tess, sa);
        ms::SetInactiveValues(&masked->mask, sa, -1);
        ms::LabelElementsWithSeeds(*reference->pmesh, seeds, tess, sb);
        ms::ConnectedLabelParameters connected;
        ms::ConstrainToConnectedCells(*masked->pmesh,
                                      MPI_COMM_WORLD,
                                      *masked->adjacency,
                                      seeds,
                                      tess,
                                      sa,
                                      connected,
                                      &context);
        ms::ConstrainToConnectedCells(
            *reference->pmesh, MPI_COMM_SELF, *reference->adjacency, seeds, tess, sb, connected);
        const auto connected_a = ByKey(*masked, sa, &masked->mask);
        EXPECT_EQ(connected_a, ByKey(*reference, sb, nullptr)) << (tets ? "tet" : "hex");
        EXPECT_EQ(connected_a.at(right_bottom), 1) << "connected";
        EXPECT_TRUE(InactiveCarry(*masked, sa, -1));

        // Geodesic, both solvers
        for (const bool graph : {false, true}) {
            ms::GeodesicParameters params;
            params.context = &context;
            const ms::GeodesicLabels ga =
                graph ? ms::LabelGeodesicGraph(*masked->pmesh, MPI_COMM_WORLD, seeds, false, params)
                      : ms::LabelGeodesicFim(*masked->pmesh, MPI_COMM_WORLD, seeds, false, params);
            const ms::GeodesicParameters plain;
            const ms::GeodesicLabels gb =
                graph
                    ? ms::LabelGeodesicGraph(*reference->pmesh, MPI_COMM_SELF, seeds, false, plain)
                    : ms::LabelGeodesicFim(*reference->pmesh, MPI_COMM_SELF, seeds, false, plain);
            const auto geodesic_a = ByKey(*masked, ga.element_seed, &masked->mask);
            EXPECT_EQ(geodesic_a, ByKey(*reference, gb.element_seed, nullptr))
                << (graph ? "graph" : "FIM");
            EXPECT_EQ(geodesic_a.at(right_bottom), 1) << (graph ? "graph" : "FIM");
            EXPECT_EQ(ga.report.unreached_vertices, gb.report.unreached_vertices);
            EXPECT_EQ(ga.report.unreached_elements, 0);
            EXPECT_EQ(ga.report.unlabeled_elements, 0);
            EXPECT_TRUE(InactiveCarry(*masked, ga.element_seed, -1));
        }
    }
}

// An empty mask (every element active) gives the same labels as no mask.
TEST(MaskLabeling, EmptyMaskChangesNothing) {
    mfem::Mesh serial = BuildMesh(false, true);
    auto run = Distribute(serial, MPI_COMM_WORLD);
    const ms::SeedSet seeds = TwoSeeds();
    ms::ElementMask empty;
    const ms::TessellationOptions tess;
    const ms::TessellationLabels a = ms::LabelTessellation(
        *run->pmesh, MPI_COMM_WORLD, seeds, tess);
    const ms::TessellationLabels b = ms::LabelTessellation(
        *run->pmesh, MPI_COMM_WORLD, seeds, tess, &empty);
    EXPECT_EQ(ByKey(*run, a.element_grain, nullptr), ByKey(*run, b.element_grain, nullptr));
    ms::MaterialContext context;
    context.mask = &empty;
    ms::GeodesicParameters params;
    params.context = &context;
    const ms::GeodesicLabels ga = ms::LabelGeodesicFim(
        *run->pmesh, MPI_COMM_WORLD, seeds, false, ms::GeodesicParameters{});
    const ms::GeodesicLabels gb = ms::LabelGeodesicFim(
        *run->pmesh, MPI_COMM_WORLD, seeds, false, params);
    EXPECT_EQ(ByKey(*run, ga.element_seed, nullptr), ByKey(*run, gb.element_seed, nullptr));
}

// Curved material (order-2 Nodes, every node warped): the geodesic modes use the nodal field on
// both meshes (AUTO) and the masked labels equal the reference, element by element; paths still
// avoid the plate. Centroid labels agree as well.
TEST(MaskLabeling, CurvedMaterialUsesTheNodalField) {
    for (const bool tets : {false, true}) {
        mfem::Mesh masked_serial = BuildMesh(false, tets);
        mfem::Mesh reference_serial = BuildMesh(true, tets);
        for (mfem::Mesh* m : {&masked_serial, &reference_serial}) {
            m->SetCurvature(2);
            m->Transform(WarpAll);
        }
        auto masked = Distribute(masked_serial, MPI_COMM_WORLD);
        auto reference = Distribute(reference_serial, MPI_COMM_SELF);
        ASSERT_FALSE(ms::IsStraightSided(*masked->pmesh, MPI_COMM_WORLD, &masked->mask));
        const ms::MaterialPointLocator locator(*masked->pmesh, false, &masked->mask);
        ms::MaterialContext context;
        context.mask = &masked->mask;
        context.locator = &locator;
        const ms::VertexField field(*masked->pmesh, ms::FieldNodes::AUTO, &masked->mask);
        EXPECT_TRUE(field.UsesMeshNodes());
        const ms::SeedSet seeds = TwoSeeds();
        const std::uint64_t right_bottom = KeyNear(*reference, 0.875, 0.25, 0.625);

        const ms::TessellationOptions tess;
        const ms::TessellationLabels a = ms::LabelTessellation(
            *masked->pmesh, MPI_COMM_WORLD, seeds, tess, &masked->mask);
        const ms::TessellationLabels b = ms::LabelTessellation(
            *reference->pmesh, MPI_COMM_SELF, seeds, tess);
        EXPECT_EQ(ByKey(*masked, a.element_seed, &masked->mask),
                  ByKey(*reference, b.element_seed, nullptr));

        for (const bool graph : {false, true}) {
            const std::string what = std::string(tets ? "tet " : "hex ") +
                                     (graph ? "graph" : "FIM");
            ms::GeodesicParameters params;
            params.context = &context;
            const ms::GeodesicParameters plain;
            const ms::GeodesicLabels ga =
                graph ? ms::LabelGeodesicGraph(*masked->pmesh, MPI_COMM_WORLD, seeds, false, params)
                      : ms::LabelGeodesicFim(*masked->pmesh, MPI_COMM_WORLD, seeds, false, params);
            const ms::GeodesicLabels gb =
                graph
                    ? ms::LabelGeodesicGraph(*reference->pmesh, MPI_COMM_SELF, seeds, false, plain)
                    : ms::LabelGeodesicFim(*reference->pmesh, MPI_COMM_SELF, seeds, false, plain);
            const auto geodesic_a = ByKey(*masked, ga.element_seed, &masked->mask);
            EXPECT_EQ(geodesic_a, ByKey(*reference, gb.element_seed, nullptr)) << what;
            EXPECT_EQ(geodesic_a.at(right_bottom), 1) << what;
            EXPECT_EQ(ga.report.unlabeled_elements, 0) << what;
            EXPECT_TRUE(InactiveCarry(*masked, ga.element_seed, -1)) << what;
        }
    }
}

// Only the plate is curved: the active elements stay straight, so the masked straightness test
// says straight (the whole mesh is curved), the geodesic field is the corner field, and the
// labels equal those of the straight active-only reference with order-2 Nodes.
TEST(MaskLabeling, CurvedPlateKeepsTheCornerField) {
    for (const bool tets : {false, true}) {
        mfem::Mesh masked_serial = BuildMesh(false, tets);
        mfem::Mesh reference_serial = BuildMesh(true, tets);
        masked_serial.SetCurvature(2);
        masked_serial.Transform(WarpPlate);
        reference_serial.SetCurvature(2);
        auto masked = Distribute(masked_serial, MPI_COMM_WORLD);
        auto reference = Distribute(reference_serial, MPI_COMM_SELF);
        EXPECT_FALSE(ms::IsStraightSided(*masked->pmesh, MPI_COMM_WORLD));
        EXPECT_TRUE(ms::IsStraightSided(*masked->pmesh, MPI_COMM_WORLD, &masked->mask));
        const ms::VertexField field(*masked->pmesh, ms::FieldNodes::AUTO, &masked->mask);
        EXPECT_FALSE(field.UsesMeshNodes());
        const ms::SeedSet seeds = TwoSeeds();
        ms::MaterialContext context;
        context.mask = &masked->mask;
        for (const bool graph : {false, true}) {
            ms::GeodesicParameters params;
            params.context = &context;
            const ms::GeodesicParameters plain;
            const ms::GeodesicLabels ga =
                graph ? ms::LabelGeodesicGraph(*masked->pmesh, MPI_COMM_WORLD, seeds, false, params)
                      : ms::LabelGeodesicFim(*masked->pmesh, MPI_COMM_WORLD, seeds, false, params);
            const ms::GeodesicLabels gb =
                graph
                    ? ms::LabelGeodesicGraph(*reference->pmesh, MPI_COMM_SELF, seeds, false, plain)
                    : ms::LabelGeodesicFim(*reference->pmesh, MPI_COMM_SELF, seeds, false, plain);
            EXPECT_EQ(ByKey(*masked, ga.element_seed, &masked->mask),
                      ByKey(*reference, gb.element_seed, nullptr))
                << (tets ? "tet " : "hex ") << (graph ? "graph" : "FIM");
        }
    }
}

/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
