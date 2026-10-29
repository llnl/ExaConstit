/**
 * @file test_adjacency_check.cpp
 * @brief Tests of conformal/adjacency_check: the grain adjacency of the
 * mesh compared with the analytic one.
 *
 * What is checked, and why:
 *  - Four Voronoi slabs (seeds on a line along x, neighbor graph 0-1-2-3), each slab two elements
 *    thick, labeled by the seeds: the mesh and the analytic graph agree (3 pairs, nothing fake or
 *    missing).
 *  - The same slabs with the first element layer of slab 2 given to grain 4 (seed 3): slabs 1 and
 *    3 now touch (fake pair (1, 3): their bisector lies inside slab 2) and slabs 1 and 2 no longer
 *    touch although their bisector runs through the mesh nodes (missing pair (1, 2)).
 *  - A grain without seed: its pairs are unchecked, not compared. Without analytic targets
 *    every pair is unchecked.
 *  - A random 30-grain Voronoi polycrystal: discrete - fake = analytic - missing, and every count
 *    and example on MPI_COMM_WORLD equals the MPI_COMM_SELF run (meshes distributed round robin),
 *    at np = 1, 2, 3, 4.
 */
#include "microstructure/conformal/adjacency_check.hpp"
#include "microstructure/conformal/node_labels.hpp"
#include "microstructure/conformal/node_targets.hpp"

#include "mfem.hpp"

#include <array>
#include <functional>
#include <memory>
#include <mpi.h>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace ms = exaconstit::microstructure;

namespace {

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

/** Element labels: 1 + the nearest seed of the element center. */
mfem::Array<int> NearestSeed(mfem::Mesh& mesh, const ms::SeedSet& s) {
    mfem::Array<int> labels(mesh.GetNE());
    mfem::Vector c(3);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.GetElementCenter(e, c);
        int best = 0;
        double best_d = HUGE_VAL;
        for (int i = 0; i < static_cast<int>(s.Size()); ++i) {
            const double d = (c(0) - s.x[i]) * (c(0) - s.x[i]) + (c(1) - s.y[i]) * (c(1) - s.y[i]) +
                             (c(2) - s.z[i]) * (c(2) - s.z[i]);
            if (d < best_d) {
                best_d = d;
                best = i;
            }
        }
        labels[e] = best + 1;
    }
    return labels;
}

/** Distributes `serial` round robin on `comm`, labels it and compares the adjacency. */
ms::AdjacencyComparison Compare(mfem::Mesh& serial,
                                MPI_Comm comm,
                                const std::function<mfem::Array<int>(mfem::Mesh&)>& labels_of,
                                const ms::SeedSet& seeds,
                                const std::vector<int>& grain_seed,
                                bool analytic = true) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> part(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        part[static_cast<std::size_t>(e)] = e % size;
    }
    mfem::ParMesh pmesh(comm, serial, part.data());
    const mfem::Array<int> labels = labels_of(pmesh);
    ms::NodeLabels node_labels(pmesh, labels);
    ms::NodeTargetOptions options;
    options.analytic = analytic;
    ms::NodeTargets targets(
        node_labels, ms::MetricType::VORONOI, seeds, ms::MetricSetup{}, grain_seed, options);
    return ms::CompareAdjacency(targets, grain_seed);
}

/** Four seeds on a line along x: four Voronoi slabs, each a quarter of the box. */
const ms::SeedSet& Slabs() {
    static const ms::SeedSet s = MakeSeeds(
        {{0.125, 0.5, 0.5}, {0.375, 0.5, 0.5}, {0.625, 0.5, 0.5}, {0.875, 0.5, 0.5}});
    return s;
}

} // namespace

// Four slabs labeled by their own seeds: the mesh graph is the analytic chain 0-1-2-3 (3 pairs),
// so nothing is fake or missing.
TEST(AdjacencyCheck, SlabsAgree) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(8, 3, 3, mfem::Element::HEXAHEDRON);
    const auto labels = [](mfem::Mesh& m) {
        return NearestSeed(m, Slabs());
    };
    const ms::AdjacencyComparison c = Compare(mesh, MPI_COMM_WORLD, labels, Slabs(), {0, 1, 2, 3});
    EXPECT_EQ(c.discrete, 3);
    EXPECT_EQ(c.analytic, 3);
    EXPECT_EQ(c.fake, 0);
    EXPECT_EQ(c.missing, 0);
    EXPECT_EQ(c.unchecked, 0);
    EXPECT_TRUE(c.fake_examples.empty());
    EXPECT_TRUE(c.missing_examples.empty());
}

// One element layer of slab 2 relabeled to grain 4: (1, 3) touch only in the mesh (their
// bisector lies inside slab 2) and (1, 2) only analytically, one fake and one missing pair.
TEST(AdjacencyCheck, FakeAndMissingPair) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(8, 3, 3, mfem::Element::HEXAHEDRON);
    // The first element layer of slab 2 (centers at x = 0.5625) goes to grain 4
    const auto labels = [](mfem::Mesh& m) {
        mfem::Array<int> l = NearestSeed(m, Slabs());
        mfem::Vector c(3);
        for (int e = 0; e < m.GetNE(); ++e) {
            m.GetElementCenter(e, c);
            if (std::fabs(c(0) - 0.5625) < 1e-9) {
                l[e] = 4;
            }
        }
        return l;
    };
    const ms::AdjacencyComparison c = Compare(mesh, MPI_COMM_WORLD, labels, Slabs(), {0, 1, 2, 3});
    EXPECT_EQ(c.discrete, 3); // (0, 1), (1, 3), (2, 3)
    EXPECT_EQ(c.analytic, 3); // (0, 1), (1, 2), (2, 3)
    EXPECT_EQ(c.fake, 1);
    EXPECT_EQ(c.missing, 1);
    ASSERT_EQ(c.fake_examples.size(), 1u);
    ASSERT_EQ(c.missing_examples.size(), 1u);
    EXPECT_EQ(c.fake_examples[0], (std::array<int, 2>{1, 3}));
    EXPECT_EQ(c.missing_examples[0], (std::array<int, 2>{1, 2}));
}

// A grain without seed has no analytic boundary to compare with, so its pair is unchecked; without
// analytic targets all three pairs are.
TEST(AdjacencyCheck, GrainsWithoutSeedAreUnchecked) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(8, 3, 3, mfem::Element::HEXAHEDRON);
    const auto labels = [](mfem::Mesh& m) {
        return NearestSeed(m, Slabs());
    };
    // Grain 4 has no seed: its pair with grain 3 is unchecked
    const ms::AdjacencyComparison c = Compare(mesh, MPI_COMM_WORLD, labels, Slabs(), {0, 1, 2, -1});
    EXPECT_EQ(c.discrete, 2);
    EXPECT_EQ(c.unchecked, 1);
    EXPECT_EQ(c.fake, 0);
    EXPECT_EQ(c.missing, 0);
    // Without analytic targets nothing is compared
    const ms::AdjacencyComparison n = Compare(
        mesh, MPI_COMM_WORLD, labels, Slabs(), {0, 1, 2, 3}, false);
    EXPECT_EQ(n.discrete, 0);
    EXPECT_EQ(n.analytic, 0);
    EXPECT_EQ(n.unchecked, 3);
}

// 30 random Voronoi grains: the counts satisfy discrete - fake = analytic - missing (both sides
// are the pairs found both ways), and every count and example equals the single-rank run.
TEST(AdjacencyCheck, PolycrystalIsPartitionIndependent) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::array<double, 3>> p(30);
    for (auto& q : p) {
        q = {u(rng), u(rng), u(rng)};
    }
    const ms::SeedSet seeds = MakeSeeds(p);
    std::vector<int> grain_seed(30);
    for (int g = 0; g < 30; ++g) {
        grain_seed[static_cast<std::size_t>(g)] = g;
    }
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(10, 10, 10, mfem::Element::HEXAHEDRON);
    const auto labels = [&](mfem::Mesh& m) {
        return NearestSeed(m, seeds);
    };
    const ms::AdjacencyComparison w = Compare(mesh, MPI_COMM_WORLD, labels, seeds, grain_seed);
    const ms::AdjacencyComparison s = Compare(mesh, MPI_COMM_SELF, labels, seeds, grain_seed);
    EXPECT_GT(w.discrete, 30);
    EXPECT_EQ(w.discrete - w.fake, w.analytic - w.missing);
    EXPECT_EQ(w.discrete, s.discrete);
    EXPECT_EQ(w.analytic, s.analytic);
    EXPECT_EQ(w.fake, s.fake);
    EXPECT_EQ(w.missing, s.missing);
    EXPECT_EQ(w.fake_examples, s.fake_examples);
    EXPECT_EQ(w.missing_examples, s.missing_examples);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::printf("polycrystal 10^3: discrete %lld analytic %lld fake %lld missing %lld\n",
                    static_cast<long long>(w.discrete),
                    static_cast<long long>(w.analytic),
                    static_cast<long long>(w.fake),
                    static_cast<long long>(w.missing));
    }
}

/** MPI-aware gtest main: MPI is initialized before the tests and finalized at exit by mfem::Mpi. */
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
