/**
 * @file test_label_repair.cpp
 * @brief Tests of conformal/label_repair: element flips that remove peninsulas and bad
 * nodes before conformal fitting.
 *
 * What is checked, and why:
 *  - Peninsula: one element of grain 1 pushed into grain 2 (one face to its grain, five to the
 *    other) flips back; the repaired labels equal the clean Voronoi labels.
 *  - Stray element: an element of grain 3 inside grain 1, far from grain 3's own cell, makes its
 *    nodes fake junctions; the repair flips it to grain 1 and no bad node is left.
 *  - Guard: a grain of one element is never emptied (the flip is blocked, its nodes stay unfit).
 *  - A flip never splits a grain: a U one element thick whose bottom lies in another grain's
 *    analytic cell stays in one piece; the flips that would cut it are undone.
 *  - Pinches are bad nodes: flips remove most of the pinches of a checkerboard around a
 *    line, and the remaining pinched nodes are held.
 *  - A peninsula in a non-analytic grain (grain_seed = -1, as for voxel maps) is still repaired,
 *    but a fixed label (a passthrough part) is never flipped into or out of.
 *  - FlipToCurrentOwner: an element whose center lies past the bisector flips to the neighbor
 *    grain beyond the hysteresis, not below it.
 *  - Rank invariance and effect: a 12-grain tet mesh with perturbed labels is repaired on
 *    MPI_COMM_WORLD (round-robin partition) and on MPI_COMM_SELF; labels and reports are
 *    identical, every peninsula is gone and fewer than half of the bad nodes remain. Run at
 *    np = 1, 2, 3, 4.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows.
 */
#include "mfem.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mpi.h>
#include <set>
#include <vector>

#include "microstructure/conformal/label_repair.hpp"
#include "microstructure/labeling/fragment_cleanup.hpp"
#include "microstructure/mesh/element_key.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

/** Point coordinates rounded to 1e-9, to match points across meshes and ranks. */
using Key = std::array<long long, 3>;

/** The Key of the center of element e. */
Key CenterKey(mfem::Mesh& mesh, int e) {
    mfem::Vector c(3);
    mesh.GetElementCenter(e, c);
    return {std::llround(c(0) * 1e9), std::llround(c(1) * 1e9), std::llround(c(2) * 1e9)};
}

/** A seed set with the points p as positions (no other columns). */
ms::SeedSet MakeSeeds(const std::vector<std::array<double, 3>>& p) {
    ms::SeedSet s;
    for (const auto& q : p) {
        s.x.push_back(q[0]);
        s.y.push_back(q[1]);
        s.z.push_back(q[2]);
    }
    return s;
}

/** Element labels: 1 + nearest seed to the element center. */
mfem::Array<int> VoronoiLabels(mfem::Mesh& mesh, const ms::SeedSet& s) {
    mfem::Array<int> labels(mesh.GetNE());
    mfem::Vector c(3);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.GetElementCenter(e, c);
        int best = 0;
        double best_d = HUGE_VAL;
        for (int i = 0; i < static_cast<int>(s.Size()); ++i) {
            const double d = std::pow(c(0) - s.x[i], 2) + std::pow(c(1) - s.y[i], 2) +
                             std::pow(c(2) - s.z[i], 2);
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }
        labels[e] = best + 1;
    }
    return labels;
}

/** Distributes `serial` over `comm` round robin (element e on rank e mod size). Collective. */
std::unique_ptr<mfem::ParMesh> Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> part(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        part[static_cast<std::size_t>(e)] = e % size;
    }
    return std::make_unique<mfem::ParMesh>(comm, serial, part.data());
}

/** The labels `base`, with the element whose center is `stray` set to `stray_label`. */
mfem::Array<int>
WithStray(mfem::Mesh& mesh, const mfem::Array<int>& base, const Key& stray, int stray_label) {
    mfem::Array<int> labels(base);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        if (CenterKey(mesh, e) == stray) {
            labels[e] = stray_label;
        }
    }
    return labels;
}

/** The Key of the point (x, y, z). */
Key KeyOf(double x, double y, double z) {
    return {std::llround(x * 1e9), std::llround(y * 1e9), std::llround(z * 1e9)};
}

/** Sum of `local` over MPI_COMM_WORLD (collective). */
int GlobalSum(int local) {
    MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return local;
}

/** Number of local elements whose label differs from `expect`. */
int Differences(const mfem::Array<int>& got, const mfem::Array<int>& expect) {
    int diff = 0;
    for (int e = 0; e < got.Size(); ++e) {
        diff += (got[e] != expect[e]) ? 1 : 0;
    }
    return GlobalSum(diff);
}

/** Face-connected pieces of grain `grain` (global). Collective. */
int Pieces(mfem::ParMesh& pmesh, const mfem::Array<int>& labels, int grain) {
    const ms::MeshAdjacency adj(pmesh);
    std::vector<std::uint64_t> component;
    ms::LabelComponents(adj, labels, ms::ComputeElementKeys(pmesh), component);
    std::set<std::uint64_t> mine;
    for (int e = 0; e < labels.Size(); ++e) {
        if (labels[e] == grain) {
            mine.insert(component[static_cast<std::size_t>(e)]);
        }
    }
    // Pieces seen on several ranks have the same id: gather the ids on every rank
    std::vector<std::uint64_t> send(mine.begin(), mine.end());
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int count = static_cast<int>(send.size());
    std::vector<int> counts(static_cast<std::size_t>(size));
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        displs[static_cast<std::size_t>(r)] = displs[static_cast<std::size_t>(r) - 1] +
                                              counts[static_cast<std::size_t>(r) - 1];
    }
    std::vector<std::uint64_t> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(send.data(),
                   count,
                   MPI_UINT64_T,
                   all.data(),
                   counts.data(),
                   displs.data(),
                   MPI_UINT64_T,
                   MPI_COMM_WORLD);
    return static_cast<int>(std::set<std::uint64_t>(all.begin(), all.end()).size());
}

} // namespace

// Grain 3 is a U one element thick inside grain 1: its arm tips reach into the analytic cell
// of its seed (above y = 0.575, the bisector of the two seeds), its bottom lies in grain 1's
// cell, so the bottom's nodes are fake junctions and its elements flip candidates. Flipping a
// bottom element would cut the U in two; such flips must be undone.
TEST(LabelRepair, FlipsNeverSplitAGrain) {
    const ms::SeedSet seeds = MakeSeeds({{0.5, 0.2, 0.5}, {0.5, 0.95, 0.5}});
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(8, 8, 8, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial, MPI_COMM_WORLD);
    mfem::Array<int> labels(pmesh->GetNE());
    mfem::Vector c(3);
    for (int e = 0; e < pmesh->GetNE(); ++e) {
        pmesh->GetElementCenter(e, c);
        const int i = static_cast<int>(c(0) * 8);
        const int j = static_cast<int>(c(1) * 8);
        const int k = static_cast<int>(c(2) * 8);
        const bool arm = (i == 1 || i == 6) && j >= 2 && j <= 6;
        const bool bottom = j == 2 && i >= 1 && i <= 6;
        labels[e] = (k == 3 && (arm || bottom)) ? 3 : 1;
    }
    EXPECT_EQ(Pieces(*pmesh, labels, 3), 1);
    const auto result = ms::RepairLabels(
        *pmesh, labels, ms::MetricType::VORONOI, seeds, {}, {0, 0, 1});
    EXPECT_GT(result.report.split_undone, 0);
    EXPECT_EQ(Pieces(*pmesh, labels, 3), 1);
    if (mfem::Mpi::Root()) {
        std::printf("U: %lld flips kept, %lld undone because they split the grain\n",
                    static_cast<long long>(result.report.flips),
                    static_cast<long long>(result.report.split_undone));
    }
}

// A 2 x 2 checkerboard around a vertical line (both grains pinch along it), without
// analytic boundaries: the pinched nodes are bad, flips remove pinches
TEST(LabelRepair, PinchesAreRemoved) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial, MPI_COMM_WORLD);
    mfem::Array<int> labels(pmesh->GetNE());
    mfem::Vector c(3);
    for (int e = 0; e < pmesh->GetNE(); ++e) {
        pmesh->GetElementCenter(e, c);
        labels[e] = ((c(0) > 0.5) != (c(1) > 0.5)) ? 2 : 1;
    }
    ms::LabelRepairOptions options;
    options.targets.analytic = false;
    const ms::SeedSet none;
    const auto result = ms::RepairLabels(
        *pmesh, labels, ms::MetricType::VORONOI, none, {}, {-1, -1}, options);
    // 7 nodes pinch at first; single flips remove most of the pinches (where two grains are
    // stacked in a checkerboard on both sides of a node it takes two flips at once, which the
    // greedy repair does not try); the remaining ones are held, none is left unfit
    EXPECT_EQ(result.report.bad_nodes_before, 7);
    EXPECT_GT(result.report.flips, 0);
    EXPECT_LT(result.labels->NumPinched(), 7);
    EXPECT_EQ(result.targets->Summary().reason[static_cast<int>(ms::TargetReason::PINCH)],
              result.labels->NumPinched());
    EXPECT_EQ(result.report.bad_nodes_after, 0);
}

// One element of grain 1 pushed into grain 2 shares one face with its grain and five with grain
// 2, a peninsula: one flip returns it, and the labels equal the clean Voronoi labels.
TEST(LabelRepair, PeninsulaFlipsBack) {
    const ms::SeedSet seeds = MakeSeeds({{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}});
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial, MPI_COMM_WORLD);
    const mfem::Array<int> clean = VoronoiLabels(*pmesh, seeds);
    mfem::Array<int> labels = WithStray(*pmesh, clean, KeyOf(7.0 / 12, 5.0 / 12, 5.0 / 12), 1);
    const auto result = ms::RepairLabels(
        *pmesh, labels, ms::MetricType::VORONOI, seeds, {}, {0, 1});
    EXPECT_EQ(Differences(labels, clean), 0);
    EXPECT_EQ(result.report.flips, 1);
    EXPECT_EQ(result.report.peninsulas_before, 1);
    EXPECT_EQ(result.report.peninsulas_after, 0);
    EXPECT_EQ(result.report.bad_nodes_after, 0);
}

// An element of grain 3 inside grain 1, far from grain 3's own cell, makes its nodes fake
// junctions; flipping it to grain 1 leaves no bad node and restores the clean labels.
TEST(LabelRepair, StrayElementFlipsToItsSurroundings) {
    const ms::SeedSet seeds = MakeSeeds({{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}, {0.9, 0.9, 0.9}});
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial, MPI_COMM_WORLD);
    const mfem::Array<int> clean = VoronoiLabels(*pmesh, seeds);
    mfem::Array<int> labels = WithStray(*pmesh, clean, KeyOf(3.0 / 12, 3.0 / 12, 3.0 / 12), 3);
    const auto result = ms::RepairLabels(
        *pmesh, labels, ms::MetricType::VORONOI, seeds, {}, {0, 1, 2});
    EXPECT_GT(result.report.bad_nodes_before, 0);
    EXPECT_EQ(result.report.bad_nodes_after, 0);
    EXPECT_EQ(Differences(labels, clean), 0);
}

// Seed 3 is far outside: its only element is the stray one
TEST(LabelRepair, OneElementGrainIsNotEmptied) {
    const ms::SeedSet seeds = MakeSeeds({{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}, {5.0, 5.0, 5.0}});
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial, MPI_COMM_WORLD);
    const mfem::Array<int> clean = VoronoiLabels(*pmesh, seeds);
    const mfem::Array<int> stray = WithStray(*pmesh, clean, KeyOf(3.0 / 12, 3.0 / 12, 3.0 / 12), 3);
    mfem::Array<int> labels(stray);
    const auto result = ms::RepairLabels(
        *pmesh, labels, ms::MetricType::VORONOI, seeds, {}, {0, 1, 2});
    EXPECT_EQ(Differences(labels, stray), 0);
    EXPECT_GT(result.report.blocked, 0);
    EXPECT_EQ(result.report.flips, 0);
    EXPECT_EQ(result.report.bad_nodes_before, 8);
    // The one-element grain has no interior node: its 8 nodes are held in the final targets
    // (under-resolved) instead of counting as unfit
    EXPECT_EQ(result.report.bad_nodes_after, 0);
    EXPECT_EQ(result.targets->Summary().under_resolved_grains, 1);
    EXPECT_EQ(result.targets->Summary().reason[static_cast<int>(ms::TargetReason::UNDER_RESOLVED)],
              8);
}

// A stray element of grain 1 inside grain 2 (a peninsula). Grain 2 without an analytic
// boundary (grain_seed -1, as for voxel maps): the peninsula is still repaired. Grain 2 as a
// fixed label (a passthrough part): nothing is flipped into or out of it.
TEST(LabelRepair, NonAnalyticAndFixedLabels) {
    const ms::SeedSet seeds = MakeSeeds({{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}});
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial, MPI_COMM_WORLD);
    const mfem::Array<int> clean = VoronoiLabels(*pmesh, seeds);
    const mfem::Array<int> stray = WithStray(*pmesh, clean, KeyOf(7.0 / 12, 5.0 / 12, 5.0 / 12), 1);
    mfem::Array<int> labels(stray);
    const auto repaired = ms::RepairLabels(
        *pmesh, labels, ms::MetricType::VORONOI, seeds, {}, {0, -1});
    EXPECT_EQ(Differences(labels, clean), 0);
    EXPECT_EQ(repaired.report.flips, 1);
    EXPECT_EQ(repaired.report.peninsulas_after, 0);

    labels = stray;
    ms::LabelRepairOptions options;
    options.labels.fixed_labels = {2};
    const auto fixed = ms::RepairLabels(
        *pmesh, labels, ms::MetricType::VORONOI, seeds, {}, {0, -1}, options);
    EXPECT_EQ(Differences(labels, stray), 0);
    EXPECT_EQ(fixed.report.flips, 0);
    EXPECT_EQ(fixed.report.peninsulas_before, 0); // a fixed label is not a peninsula grain
}

// An element of grain 1 whose center lies half an element past the bisector, inside grain 2:
// it flips to grain 2 (its face neighbors there grew from seed 1), and only once
TEST(LabelRepair, FlipToCurrentOwner) {
    const ms::SeedSet seeds = MakeSeeds({{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}});
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial, MPI_COMM_WORLD);
    const mfem::Array<int> clean = VoronoiLabels(*pmesh, seeds);
    mfem::Array<int> labels = WithStray(*pmesh, clean, KeyOf(7.0 / 12, 5.0 / 12, 5.0 / 12), 1);
    std::vector<char> flipped;
    EXPECT_EQ(ms::FlipToCurrentOwner(
                  *pmesh, labels, ms::MetricType::VORONOI, seeds, {}, {0, 1}, 0.2, flipped),
              1);
    EXPECT_EQ(Differences(labels, clean), 0);
    // A hysteresis beyond the element's offset (0.5 h) flips nothing
    mfem::Array<int> again = WithStray(*pmesh, clean, KeyOf(7.0 / 12, 5.0 / 12, 5.0 / 12), 1);
    std::vector<char> fresh;
    EXPECT_EQ(ms::FlipToCurrentOwner(
                  *pmesh, again, ms::MetricType::VORONOI, seeds, {}, {0, 1}, 0.6, fresh),
              0);
}

// 12 seeds from a fixed pseudo-random sequence; labels perturbed on about 1 element in 10 by
// taking the label of the next element in the serial order (a deterministic, messy start)
TEST(LabelRepair, RankInvariance) {
    std::vector<std::array<double, 3>> p;
    double r = 0.1234;
    const auto next = [&r]() {
        r = std::fmod(r * 9301.0 + 0.49297, 1.0);
        return 0.05 + 0.9 * r;
    };
    for (int i = 0; i < 12; ++i) {
        const double x = next(), y = next(), z = next();
        p.push_back({x, y, z});
    }
    const ms::SeedSet seeds = MakeSeeds(p);
    std::vector<int> grain_seed(12);
    for (int i = 0; i < 12; ++i) {
        grain_seed[static_cast<std::size_t>(i)] = i;
    }
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::TETRAHEDRON);
    mfem::Array<int> serial_labels = VoronoiLabels(serial, seeds);
    std::map<Key, int> perturbed;
    for (int e = 0; e < serial.GetNE(); ++e) {
        int label = serial_labels[e];
        if (e % 10 == 3 && e + 1 < serial.GetNE()) {
            label = serial_labels[e + 1];
        }
        perturbed[CenterKey(serial, e)] = label;
    }
    const auto labels_on = [&](mfem::ParMesh& m) {
        mfem::Array<int> labels(m.GetNE());
        for (int e = 0; e < m.GetNE(); ++e) {
            labels[e] = perturbed.at(CenterKey(m, e));
        }
        return labels;
    };
    auto dist = Distribute(serial, MPI_COMM_WORLD);
    auto self = Distribute(serial, MPI_COMM_SELF);
    mfem::Array<int> ld = labels_on(*dist);
    mfem::Array<int> ls = labels_on(*self);
    const auto rd =
        ms::RepairLabels(*dist, ld, ms::MetricType::VORONOI, seeds, {}, grain_seed).report;
    const auto rs =
        ms::RepairLabels(*self, ls, ms::MetricType::VORONOI, seeds, {}, grain_seed).report;
    std::map<Key, int> self_label;
    for (int e = 0; e < self->GetNE(); ++e) {
        self_label[CenterKey(*self, e)] = ls[e];
    }
    int diff = 0;
    for (int e = 0; e < dist->GetNE(); ++e) {
        diff += (self_label.at(CenterKey(*dist, e)) != ld[e]) ? 1 : 0;
    }
    EXPECT_EQ(GlobalSum(diff), 0);
    EXPECT_EQ(rd.passes, rs.passes);
    EXPECT_EQ(rd.flips, rs.flips);
    EXPECT_EQ(rd.undone, rs.undone);
    EXPECT_EQ(rd.bad_nodes_after, rs.bad_nodes_after);
    EXPECT_EQ(rd.peninsulas_after, rs.peninsulas_after);
    // The repair removes every peninsula and most bad nodes of this coarse 12-grain mesh (the
    // rest are mostly triple lines that end on box edges, which no flip can avoid)
    EXPECT_GT(rd.peninsulas_before, 0);
    EXPECT_EQ(rd.peninsulas_after, 0);
    EXPECT_LT(2 * rd.bad_nodes_after, rd.bad_nodes_before);
}

/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
