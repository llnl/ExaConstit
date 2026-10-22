/**
 * @file test_labeling.cpp
 * @brief Tests of element labeling, grain-id compaction, orientation assignment and the
 * GenerateMicrostructure pipeline (labeling/, orientation/, microstructure_driver.cpp).
 *
 * Every test runs on all ranks: the labeler's compaction step and the driver are collective
 * (every rank of the communicator must call them, because the ranks exchange data inside).
 * Rank-count independence is checked inside each run: the distributed result over
 * MPI_COMM_WORLD is gathered into serial element order and compared with a reference every rank
 * computes alone on the whole serial mesh (MPI_COMM_SELF), which is exactly the np = 1 answer.
 * Running the executable at np = 1..4 (test_labeling.cmake) therefore compares np = 2, 3, 4
 * against np = 1.
 *
 * Meshes: test/data/cube_tet.mesh (750 tets in the unit cube) and the 8 x 8 x 8 hex box that
 * ExaConstit builds for [Mesh] type = "auto" (exaconstit::MakeAutoHexMesh).
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever in that
 * collective call.
 */
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/labeling/tessellation_labeler.hpp"
#include "microstructure/mesh/mesh_sampler.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_driver.hpp"
#include "microstructure/orientation/orientation_assigner.hpp"
#include "microstructure/orientation/random_orientation.hpp"
#include "microstructure/util/counter_rng.hpp"
#include "utilities/auto_mesh.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

const fs::path data_dir = fs::path(MS_LABELING_DATA_DIR);

/** test/data/cube_tet.mesh, read the way ExaConstit reads a [Mesh] type = "file" mesh. */
mfem::Mesh CubeTet() {
    return mfem::Mesh((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
}

/** ExaConstit's auto box: n x n x n hexes in the unit cube, lexicographic element order. */
mfem::Mesh AutoHex(int n) {
    return exaconstit::MakeAutoHexMesh({n, n, n}, {1.0, 1.0, 1.0});
}

/**
 * n random seeds in the unit cube from the counter-based RNG (purpose GENERIC), so every rank
 * builds the identical set. source_id = 3 i + 1 on purpose: ids that are not 0..N-1 show that
 * orientations are keyed by the source id and not by the seed or grain index. With
 * `max_weight` > 0 each seed also gets a weight uniform in [0, max_weight) (Laguerre).
 */
ms::SeedSet RandomSeeds(int n, std::uint64_t key, double max_weight = 0.0) {
    ms::SeedSet seeds;
    for (int i = 0; i < n; ++i) {
        const ms::CounterRng rng(key, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        seeds.x.push_back(rng.UniformDouble(0u));
        seeds.y.push_back(rng.UniformDouble(1u));
        seeds.z.push_back(rng.UniformDouble(2u));
        if (max_weight > 0.0) {
            seeds.weight.push_back(max_weight * rng.UniformDouble(3u));
        }
        seeds.source_id.push_back(3 * static_cast<std::int64_t>(i) + 1);
    }
    return seeds;
}

/** The distributed mesh with the serial-order bookkeeping needed to gather results. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** Partitions `serial` over `comm` (collective), keeping local -> serial element ids. */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/**
 * Brute-force nearest seed of every local element centroid: loop over all seeds with the same
 * metric functor the search uses (the SeedIndex is only used for its device copy of the seeds and
 * the functor; with SEQ_EXEC those are host pointers). Strict `<` keeps the lowest index on ties,
 * the documented tie-break.
 */
template <typename Metric>
std::vector<int> BruteForceLabels(mfem::Mesh& mesh, const ms::SeedSet& seeds,
                                  const ms::MetricSetup& setup) {
    ms::SeedIndex<axom::SEQ_EXEC, Metric> index;
    index.Build(seeds, setup, 1, axom::execution_space<axom::SEQ_EXEC>::allocatorID());
    const Metric& metric = index.GetMetric();

    ms::ElementSamples samples;
    ms::ComputeElementSamples(mesh, -1, samples);
    const int ne = mesh.GetNE();
    std::vector<int> labels(static_cast<std::size_t>(ne), -1);
    if (ne == 0) {
        return labels;
    }
    const double* cx = samples.cx.HostRead();
    const double* cy = samples.cy.HostRead();
    const double* cz = samples.cz.HostRead();
    for (int e = 0; e < ne; ++e) {
        double best_f = 0.0;
        int best_id = -1;
        for (int i = 0; i < static_cast<int>(seeds.Size()); ++i) {
            const double f = metric.Eval(i, cx[e], cy[e], cz[e]);
            if (best_id < 0 || f < best_f) {
                best_f = f;
                best_id = i;
            }
        }
        labels[static_cast<std::size_t>(e)] = best_id;
    }
    return labels;
}

/**
 * A partition of `serial` over `comm` in which the last rank owns no element (when there are at
 * least 2 ranks): elements are dealt round-robin to ranks 0 .. nranks-2. Built by hand instead of
 * with METIS, so the empty-rank code paths (sampler, labeler, compaction, seeding) are exercised.
 * Collective (MakeParMesh).
 */
Distributed DistributeWithEmptyLastRank(mfem::Mesh& serial, MPI_Comm comm) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const int owners = (nranks > 1) ? nranks - 1 : 1;
    Distributed d;
    d.order.partitioning.SetSize(serial.GetNE());
    int num_local = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        d.order.partitioning[e] = e % owners;
        num_local += (e % owners == rank) ? 1 : 0;
    }
    d.order.local_to_serial.SetSize(num_local);
    int next = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        if (d.order.partitioning[e] == rank) {
            d.order.local_to_serial[next++] = e;
        }
    }
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** Options of the "-n N" quick path with N grains and the given RNG key. */
ms::MicrostructureOptions QuickOptions(int num_grains, std::uint64_t rng_seed) {
    ms::MicrostructureOptions options;
    options.seeds.num_seeds = num_grains;
    options.seeds.rng_seed = rng_seed;
    return options;
}

} // namespace

// The labeler's exact search must give every element exactly the seed a loop over all seeds
// gives (same metric functor, lowest index on ties), on the tet mesh and the 8^3 hex box, for
// all four metrics: Voronoi; Laguerre with weights (planar boundaries shifted toward the seed
// with the smaller weight, so the search must use its weight-aware lower bounds); Additive with
// weights (curved, hyperbolic boundaries); and Quadric with a non-diagonal global tensor
// (elongated grains; the bound uses the tensor's smallest eigenvalue). Checked per local element
// at any rank count; both sides use the same element centroids and metric setup.
TEST(TessellationLabeler, MatchesBruteForce) {
    struct Case {
        const char* name;
        bool tets;
        int num_seeds;
        ms::MetricType metric;
        double max_weight;
    };
    const Case cases[] = {{"cube_tet voronoi", true, 37, ms::MetricType::VORONOI, 0.0},
                          {"hex8 voronoi", false, 50, ms::MetricType::VORONOI, 0.0},
                          {"cube_tet laguerre", true, 37, ms::MetricType::LAGUERRE, 0.02},
                          {"hex8 laguerre", false, 50, ms::MetricType::LAGUERRE, 0.02},
                          {"cube_tet additive", true, 37, ms::MetricType::ADDITIVE, 0.05},
                          {"hex8 additive", false, 50, ms::MetricType::ADDITIVE, 0.05},
                          {"cube_tet quadric", true, 37, ms::MetricType::QUADRIC, 0.01},
                          {"hex8 quadric", false, 50, ms::MetricType::QUADRIC, 0.01}};
    // Symmetric positive definite tensor [Axx, Ayy, Azz, Ayz, Axz, Axy] (diagonally dominant)
    const std::array<double, 6> tensor = {2.0, 1.0, 0.5, 0.1, 0.0, 0.2};
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        mfem::Mesh serial = c.tets ? CubeTet() : AutoHex(8);
        Distributed dist = Distribute(serial, MPI_COMM_WORLD);
        const ms::SeedSet seeds = RandomSeeds(c.num_seeds, 2024u, c.max_weight);

        ms::TessellationOptions tess;
        tess.metric = c.metric;
        tess.global_tensor = tensor;
        mfem::Array<int> element_seed;
        ms::LabelElementsWithSeeds(*dist.pmesh, seeds, tess, element_seed);

        ms::MetricSetup setup;
        setup.global_tensor = tensor;
        std::vector<int> expected;
        switch (c.metric) {
        case ms::MetricType::VORONOI:
            expected = BruteForceLabels<ms::VoronoiMetric>(*dist.pmesh, seeds, setup);
            break;
        case ms::MetricType::LAGUERRE:
            expected = BruteForceLabels<ms::LaguerreMetric>(*dist.pmesh, seeds, setup);
            break;
        case ms::MetricType::ADDITIVE:
            expected = BruteForceLabels<ms::AdditiveMetric>(*dist.pmesh, seeds, setup);
            break;
        case ms::MetricType::QUADRIC:
            expected = BruteForceLabels<ms::QuadricMetric>(*dist.pmesh, seeds, setup);
            break;
        }
        EXPECT_EQ(element_seed.Size(), dist.pmesh->GetNE());
        const int* got = (element_seed.Size() > 0) ? element_seed.HostRead() : nullptr;
        for (int e = 0; e < element_seed.Size(); ++e) {
            EXPECT_EQ(got[e], expected[static_cast<std::size_t>(e)]) << "local element " << e;
        }
    }
}

// Compaction must number exactly the seeds that won elements, 1..N with no gaps, in increasing
// seed order. 200 seeds on a 4^3 = 64 element box guarantee that most seeds win nothing, and seed
// 6 is placed exactly on seed 5. The test uses the (default) Voronoi metric, where a coincident
// seed has exactly the same f as the lower-index one and so never wins (ties go to the lower
// index); with weighted metrics a coincident seed with a larger weight could win. The expected
// set of used seeds is computed independently of the bitmask code: a per-seed 0/1 array reduced
// with MPI_MAX.
TEST(TessellationLabeler, CompactIdsHaveNoGaps) {
    mfem::Mesh serial = AutoHex(4);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    ms::SeedSet seeds = RandomSeeds(200, 7u);
    seeds.x[6] = seeds.x[5];
    seeds.y[6] = seeds.y[5];
    seeds.z[6] = seeds.z[5];

    const ms::TessellationLabels labels =
        ms::LabelTessellation(*dist.pmesh, MPI_COMM_WORLD, seeds, ms::TessellationOptions{});
    const int ne = dist.pmesh->GetNE();
    EXPECT_EQ(labels.num_coincident_seeds, 1);

    // Independent reference: which seeds won at least one element anywhere
    std::vector<int> used(seeds.Size(), 0);
    const int* seed_of = (ne > 0) ? labels.element_seed.HostRead() : nullptr;
    for (int e = 0; e < ne; ++e) {
        used[static_cast<std::size_t>(seed_of[e])] = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, used.data(), static_cast<int>(used.size()), MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    int expected_grains = 0;
    for (const int u : used) {
        expected_grains += u;
    }
    EXPECT_EQ(labels.num_grains, expected_grains);
    EXPECT_LE(labels.num_grains, 64); // at most one grain per element
    EXPECT_EQ(used[6], 0);            // the coincident seed never wins (Voronoi: equal f)

    // Grain g <-> seed tables: grains in increasing seed order, unused seeds have grain 0
    EXPECT_EQ(static_cast<int>(labels.grain_seed.size()), labels.num_grains);
    int expected_next = 1;
    for (std::size_t s = 0; s < seeds.Size(); ++s) {
        if (used[s] != 0) {
            EXPECT_EQ(labels.seed_grain[s], expected_next);
            if (expected_next <= static_cast<int>(labels.grain_seed.size())) {
                EXPECT_EQ(labels.grain_seed[static_cast<std::size_t>(expected_next - 1)],
                          static_cast<int>(s));
            }
            ++expected_next;
        } else {
            EXPECT_EQ(labels.seed_grain[s], 0);
        }
    }

    // Every element carries the grain of its seed, and every grain 1..N owns an element
    std::vector<int> grain_count(static_cast<std::size_t>(labels.num_grains) + 1, 0);
    const int* grain_of = (ne > 0) ? labels.element_grain.HostRead() : nullptr;
    for (int e = 0; e < ne; ++e) {
        EXPECT_EQ(grain_of[e], labels.seed_grain[static_cast<std::size_t>(seed_of[e])]);
        const bool in_range = (grain_of[e] >= 1 && grain_of[e] <= labels.num_grains);
        EXPECT_TRUE(in_range);
        if (in_range) {
            grain_count[static_cast<std::size_t>(grain_of[e])] += 1;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, grain_count.data(), static_cast<int>(grain_count.size()),
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for (int g = 1; g <= labels.num_grains; ++g) {
        EXPECT_GT(grain_count[static_cast<std::size_t>(g)], 0) << "grain " << g;
    }
}

// An element whose centroid is not finite gets no seed from the search (FindBest returns id1 = -1:
// every comparison with NaN is false). The labeler must turn that into one clear error, thrown on
// every rank together, naming the element, instead of a misleading compaction error. Vertex 0 of
// the 2x2x2 box is a corner of element 0 only; setting its x to NaN makes that element's centroid
// NaN (the centroid is a quadrature sum over the element's node coordinates).
TEST(TessellationLabeler, NonFiniteCentroidThrowsOnAllRanks) {
    mfem::Mesh serial = AutoHex(2);
    serial.GetVertex(0)[0] = std::numeric_limits<double>::quiet_NaN();
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::SeedSet seeds = RandomSeeds(5, 11u);
    bool threw = false;
    try {
        ms::LabelTessellation(*dist.pmesh, MPI_COMM_WORLD, seeds, ms::TessellationOptions{});
    } catch (const std::runtime_error& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("non-finite element centroid"), std::string::npos)
            << e.what();
    }
    EXPECT_TRUE(threw);
}

// A seed index outside [0, N) on one rank only (the last) must make every rank throw, because
// the check is reduced over all ranks before anything else happens; otherwise the ranks without
// the bad index would hang in the next collective call.
TEST(TessellationLabeler, BadSeedIndexThrowsOnAllRanks) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    mfem::Array<int> element_seed(3);
    element_seed = 0;
    if (rank == nranks - 1) {
        element_seed[1] = 5; // only seeds 0..4 exist
    }
    ms::TessellationLabels labels;
    EXPECT_THROW(ms::CompactSeedLabels(MPI_COMM_WORLD, element_seed, 5, labels),
                 std::invalid_argument);
}

// The full pipeline (preflight, seeding, labeling, cleanup, orientations) must give the same
// grains for every rank count: the labels gathered into serial element order, the grain count,
// the orientations (bitwise) and the source ids must equal the single-rank run on the whole mesh.
// Cases: cube_tet.mesh with 20 grains (rng_seed 42) and the 8^3 auto box with 8 grains, the two
// end-to-end examples. Ids must be compact (1..N, every grain used), orientations unit
// quaternions with w >= 0 keyed by the grain's source id, and every grain in region 1.
TEST(GenerateMicrostructure, RankCountIndependent) {
    struct Case {
        const char* name;
        bool tets;
        int num_grains;
        std::uint64_t rng_seed;
        std::int64_t tiny_fragments; ///< edge-connected single elements (measured, see below)
    };
    const Case cases[] = {{"cube_tet 20 grains", true, 20, 42u, 6},
                          {"hex8 8 grains", false, 8, 0u, 1}};
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        mfem::Mesh serial = c.tets ? CubeTet() : AutoHex(8);
        const ms::MicrostructureOptions options = QuickOptions(c.num_grains, c.rng_seed);

        // Distributed run, gathered to rank 0 in serial element order
        Distributed dist = Distribute(serial, MPI_COMM_WORLD);
        ms::GenerationReport report;
        ms::MicrostructureResult result =
            ms::GenerateMicrostructure(options, *dist.pmesh, MPI_COMM_WORLD, &report);
        result.element_grain.HostRead();
        const std::vector<int> gathered =
            ms::GatherToSerial(MPI_COMM_WORLD, dist.order, result.element_grain, 0);

        // Single-rank reference on the whole mesh (every rank computes it on its own)
        Distributed alone = Distribute(serial, MPI_COMM_SELF);
        ms::MicrostructureResult reference =
            ms::GenerateMicrostructure(options, *alone.pmesh, MPI_COMM_SELF);

        // The tables are replicated, so every rank compares them
        EXPECT_EQ(result.num_grains, reference.num_grains);
        EXPECT_EQ(result.grain_quat, reference.grain_quat);
        EXPECT_EQ(result.grain_source_id, reference.grain_source_id);
        EXPECT_EQ(report.num_grains, result.num_grains);
        EXPECT_EQ(report.num_elements, static_cast<std::int64_t>(serial.GetNE()));
        // 8 or 20 seeds on 512 / 750 elements: every seed wins elements in these cases. With
        // centroid labels some convex Voronoi cells leave a single element that touches the rest
        // of its grain only along an edge (1 such piece on hex8, 6 on cube_tet); the default
        // cleanup ("auto": merge tiny pieces, split large ones) merges them, so the final count
        // is the seed count again and nothing is split.
        EXPECT_EQ(report.num_labeled_grains, c.num_grains);
        EXPECT_EQ(result.num_grains, c.num_grains);
        EXPECT_EQ(report.cleanup.fragments_found, c.tiny_fragments);
        EXPECT_EQ(report.cleanup.fragments_merged, c.tiny_fragments);
        EXPECT_EQ(report.cleanup.fragments_split, 0);
        EXPECT_EQ(result.num_regions, 1);
        EXPECT_EQ(result.grain_region,
                  std::vector<int>(static_cast<std::size_t>(result.num_grains), 1));

        // Orientations: unit norm, canonical sign, keyed by (rng_seed, ORIENTATION, source id)
        for (int g = 0; g < result.num_grains; ++g) {
            const double* q = &result.grain_quat[4 * static_cast<std::size_t>(g)];
            const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            EXPECT_NEAR(norm, 1.0, 1e-14);
            EXPECT_GE(q[0], 0.0);
            double expected[4];
            const ms::CounterRng rng(
                c.rng_seed, ms::RngPurpose::ORIENTATION,
                static_cast<std::uint64_t>(result.grain_source_id[static_cast<std::size_t>(g)]));
            ms::RandomQuaternion(rng, 0u, expected);
            // The driver normalizes every row on the host: expected / |expected|
            const double n = std::sqrt(expected[0] * expected[0] + expected[1] * expected[1] +
                                       expected[2] * expected[2] + expected[3] * expected[3]);
            for (int k = 0; k < 4; ++k) {
                EXPECT_EQ(q[k], expected[k] / n);
            }
        }

        // Labels: identical to the single-rank run, element by element, and compact
        if (rank == 0) {
            const int ne = serial.GetNE();
            EXPECT_EQ(static_cast<int>(gathered.size()), ne);
            const int* ref = reference.element_grain.HostRead();
            std::vector<int> count(static_cast<std::size_t>(reference.num_grains) + 1, 0);
            for (int e = 0; e < ne && e < static_cast<int>(gathered.size()); ++e) {
                EXPECT_EQ(gathered[static_cast<std::size_t>(e)], ref[e]) << "serial element " << e;
                if (ref[e] >= 1 && ref[e] <= reference.num_grains) {
                    count[static_cast<std::size_t>(ref[e])] += 1;
                }
            }
            for (int g = 1; g <= reference.num_grains; ++g) {
                EXPECT_GT(count[static_cast<std::size_t>(g)], 0) << "grain " << g;
            }
        }
    }
}

// A rank that owns no element must still take part in every collective step and get the same
// replicated tables: the partition is forced so that the last rank is empty (np >= 2), which
// exercises the ne == 0 branches of the element sampler, the labeler (it still builds the seed
// index), the compaction (an empty local bitmask) and the seeding. The labels gathered into serial
// order must equal the single-rank run, as for any other partition. At np = 1 the case reduces to
// the ordinary single-rank run.
TEST(GenerateMicrostructure, EmptyRankGivesSameGrains) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    mfem::Mesh serial = AutoHex(4);
    const ms::MicrostructureOptions options = QuickOptions(6, 3u);

    Distributed dist = DistributeWithEmptyLastRank(serial, MPI_COMM_WORLD);
    if (nranks > 1 && rank == nranks - 1) {
        EXPECT_EQ(dist.pmesh->GetNE(), 0);
    }
    ms::MicrostructureResult result =
        ms::GenerateMicrostructure(options, *dist.pmesh, MPI_COMM_WORLD);
    EXPECT_EQ(result.element_grain.Size(), dist.pmesh->GetNE());
    result.element_grain.HostRead();
    const std::vector<int> gathered =
        ms::GatherToSerial(MPI_COMM_WORLD, dist.order, result.element_grain, 0);

    Distributed alone = Distribute(serial, MPI_COMM_SELF);
    ms::MicrostructureResult reference =
        ms::GenerateMicrostructure(options, *alone.pmesh, MPI_COMM_SELF);
    EXPECT_EQ(result.num_grains, reference.num_grains);
    EXPECT_EQ(result.grain_quat, reference.grain_quat);
    if (rank == 0) {
        const int* ref = reference.element_grain.HostRead();
        EXPECT_EQ(static_cast<int>(gathered.size()), serial.GetNE());
        for (int e = 0; e < serial.GetNE() && e < static_cast<int>(gathered.size()); ++e) {
            EXPECT_EQ(gathered[static_cast<std::size_t>(e)], ref[e]) << "serial element " << e;
        }
    }
}

// Orientation assignment on its own. Grain g grew from seed grain_seed[g-1]; with source ids
// {10, 3, 77, 5} and grain_seed {0, 2, 3}, the grains' source ids are {10, 77, 5}.
//  - "random": each quaternion is RandomQuaternion of stream (rng_seed, ORIENTATION, source id)
//    divided by its norm (bitwise), unit norm, w >= 0. Keying by source id means a grain keeps
//    its orientation when the grain numbering changes: with grain_seed {2} alone, grain 1 gets
//    the quaternion grain 2 had.
//  - "from_source": the seeds' quaternions are normalized to unit length and otherwise passed
//    through (no sign change, reordering or convention conversion): (0, -1, 0, 0) keeps its
//    sign, and the norm-2 row (2, 0, 0, 0) becomes (1, 0, 0, 0).
//  - "list" (not implemented) and "from_source" without quaternions throw.
TEST(OrientationAssigner, RandomKeyedBySourceIdAndPassThrough) {
    ms::SeedSet seeds;
    seeds.x = {0.1, 0.2, 0.3, 0.4};
    seeds.y = seeds.x;
    seeds.z = seeds.x;
    seeds.source_id = {10, 3, 77, 5};
    const std::vector<int> grain_seed = {0, 2, 3};
    const std::uint64_t rng_seed = 99u;

    ms::OrientationOptions random;
    ms::MicrostructureResult result;
    ms::AssignOrientations(random, rng_seed, seeds, grain_seed, result);
    EXPECT_EQ(result.num_grains, 3);
    EXPECT_EQ(result.grain_source_id, (std::vector<std::int64_t>{10, 77, 5}));
    ASSERT_EQ(result.grain_quat.size(), 12u); // no collective call follows in this test
    for (int g = 0; g < 3; ++g) {
        double expected[4];
        ms::RandomQuaternion(
            ms::CounterRng(rng_seed, ms::RngPurpose::ORIENTATION,
                           static_cast<std::uint64_t>(result.grain_source_id[g])),
            0u, expected);
        double norm2 = 0.0;
        for (int k = 0; k < 4; ++k) {
            norm2 += expected[k] * expected[k];
        }
        EXPECT_NEAR(std::sqrt(norm2), 1.0, 1e-14);
        // Every row is normalized on the host: expected / |expected|, bitwise
        for (int k = 0; k < 4; ++k) {
            EXPECT_EQ(result.grain_quat[4 * g + k], expected[k] / std::sqrt(norm2));
        }
        EXPECT_GE(result.grain_quat[4 * g], 0.0);
    }

    // Renumbering: seed 2 alone becomes grain 1 and keeps its orientation
    ms::MicrostructureResult renumbered;
    ms::AssignOrientations(random, rng_seed, seeds, std::vector<int>{2}, renumbered);
    for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(renumbered.grain_quat[k], result.grain_quat[4 + k]);
    }

    // Pass-through, normalized to unit length
    seeds.quat = {1.0, 0.0, 0.0, 0.0,  0.5, 0.5, 0.5, 0.5,
                  0.0, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0};
    ms::OrientationOptions from_source;
    from_source.source = "from_source";
    ms::MicrostructureResult passed;
    ms::AssignOrientations(from_source, rng_seed, seeds, grain_seed, passed);
    EXPECT_EQ(passed.grain_quat,
              (std::vector<double>{1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0}));

    // Not implemented / missing data
    ms::OrientationOptions list;
    list.source = "list";
    EXPECT_THROW(ms::AssignOrientations(list, rng_seed, seeds, grain_seed, passed),
                 std::invalid_argument);
    seeds.quat.clear();
    EXPECT_THROW(ms::AssignOrientations(from_source, rng_seed, seeds, grain_seed, passed),
                 std::invalid_argument);
}

// Options the pipeline cannot honor must be refused, on every rank together and before any
// collective work, instead of being silently ignored:
// - Regions.mode = "from_phase" with random seeds: random seeds carry no phases, so there is
//   nothing to build regions from (regions from phases are implemented since Stage 1d, for seed
//   tables and voxel maps; see test_driver_stage1d);
// - Seeds.source = "file" without Seeds.file: seed tables are implemented since Stage 1d, but
//   the driver refuses a table source that names no table;
// - connected / geodesic distance (not implemented; the labeler's CheckSupportedTessellation,
//   which the driver calls before seeding), "list" orientations (not implemented) and a vote
//   level outside 0..3.
// (Fragment splitting, quadrature voting and periodic grains are implemented since Stage 1b; see
// test_driver_stage1bc.)
TEST(GenerateMicrostructure, UnsupportedOptionsThrowOnAllRanks) {
    mfem::Mesh serial = AutoHex(4);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);

    auto throws_with = [&](auto&& mutate) {
        ms::MicrostructureOptions options = QuickOptions(4, 1u);
        mutate(options);
        try {
            ms::GenerateMicrostructure(options, *dist.pmesh, MPI_COMM_WORLD);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    using O = ms::MicrostructureOptions;
    EXPECT_TRUE(throws_with([](O& o) { o.regions.mode = "from_phase"; }));
    EXPECT_TRUE(throws_with([](O& o) {
        o.seeds.source = "file";
        o.seeds.placement = ms::SeedPlacement::FILE;
    }));
    EXPECT_TRUE(throws_with([](O& o) { o.tessellation.distance = ms::DistanceMode::CONNECTED; }));
    EXPECT_TRUE(throws_with([](O& o) { o.tessellation.distance = ms::DistanceMode::GEODESIC; }));
    EXPECT_TRUE(throws_with([](O& o) { o.orientations.source = "list"; }));
    EXPECT_TRUE(throws_with([](O& o) { o.tessellation.vote_levels = 4; }));
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
