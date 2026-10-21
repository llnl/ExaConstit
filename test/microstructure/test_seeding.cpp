#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/seeding/min_distance_grid.hpp"
#include "microstructure/seeding/seed_generators.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Tests for seed generation. Every test runs on all ranks (GenerateSeeds is collective).
 *
 * Rank-count independence is checked inside each run: the distributed result over
 * MPI_COMM_WORLD is compared bitwise with a reference that every rank computes alone on the
 * whole serial mesh (MPI_COMM_SELF), which is exactly the np = 1 answer. Running the executable at
 * np = 1..4 (test_seeding.cmake) therefore checks np = 2, 3, 4 against np = 1.
 *
 * Meshes:
 *  - unit cube [0,1]^3 of hexes or tets (MakeCartesian3D);
 *  - an L-shaped block: [0,2] x [0,2] x [0,1] without the notch [1,2] x [1,2] x [0,1], built cell
 *    by cell from hexes or from 6 tets per cell. It is non-convex, so rejection sampling must
 *    really reject candidates (a quarter of the box is outside the material).
 */

namespace ms = exaconstit::microstructure;

namespace {

/** Unit cube [0,1]^3 with n^3 cells (each split into 6 tets for TETRAHEDRON). */
mfem::Mesh UnitCube(int n, mfem::Element::Type type) {
    return mfem::Mesh::MakeCartesian3D(n, n, n, type, 1.0, 1.0, 1.0, false);
}

/**
 * L-shaped block with cells of size 1/n: [0,2] x [0,2] x [0,1] minus the notch
 * [1,2] x [1,2] x [0,1] (volume 3). Hexes, or the 6-tet Kuhn split of every cell (the 6 tets
 * along the cell diagonal from its (0,0,0) to its (1,1,1) corner; the split is the same in every
 * cell, so neighboring cells share face diagonals and the mesh is conforming).
 */
mfem::Mesh LShape(int n, bool tets) {
    const int nx = 2 * n;
    const int ny = 2 * n;
    const int nz = n;
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    auto kept = [&](int i, int j) { return !(i >= n && j >= n); };

    int num_cells = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            num_cells += kept(i, j) ? nz : 0;
        }
    }
    mfem::Mesh mesh(3, (nx + 1) * (ny + 1) * (nz + 1), tets ? 6 * num_cells : num_cells, 0, 3);
    for (int k = 0; k <= nz; k++) {
        for (int j = 0; j <= ny; j++) {
            for (int i = 0; i <= nx; i++) {
                mesh.AddVertex(double(i) / n, double(j) / n, double(k) / n);
            }
        }
    }
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (!kept(i, j)) {
                    continue;
                }
                // Corner c = bx + 2 by + 4 bz of the cell
                int c[8];
                for (int b = 0; b < 8; b++) {
                    c[b] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                }
                if (!tets) {
                    const int hex[8] = {c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]};
                    mesh.AddHex(hex);
                } else {
                    // Kuhn split: one tet per axis order (a, b, c) of the path 000 -> 111
                    const int orders[6][3] = {{1, 2, 4}, {1, 4, 2}, {2, 1, 4},
                                              {2, 4, 1}, {4, 1, 2}, {4, 2, 1}};
                    for (const auto& o : orders) {
                        const int tet[4] = {c[0], c[o[0]], c[o[0] + o[1]], c[7]};
                        mesh.AddTet(tet);
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

/** True if p lies in the closed L-shaped block (with a round-off tolerance). */
bool InLShape(double x, double y, double z) {
    const double tol = 1e-12;
    const bool in_block = x >= -tol && x <= 2 + tol && y >= -tol && y <= 2 + tol && z >= -tol &&
                          z <= 1 + tol;
    const bool in_notch = x > 1 + tol && y > 1 + tol;
    return in_block && !in_notch;
}

/** Distributes `serial` over MPI_COMM_WORLD (collective). */
std::unique_ptr<mfem::ParMesh> Distribute(mfem::Mesh& serial) {
    const ms::SerialOrder order = ms::BuildSerialOrder(serial, MPI_COMM_WORLD);
    return ms::MakeParMesh(MPI_COMM_WORLD, serial, order);
}

/** Bitwise equality of positions and source ids of two seed sets. */
void ExpectSameSeeds(const ms::SeedSet& a, const ms::SeedSet& b) {
    ASSERT_EQ(a.Size(), b.Size());
    for (std::size_t i = 0; i < a.Size(); i++) {
        EXPECT_EQ(a.x[i], b.x[i]) << "seed " << i;
        EXPECT_EQ(a.y[i], b.y[i]) << "seed " << i;
        EXPECT_EQ(a.z[i], b.z[i]) << "seed " << i;
        EXPECT_EQ(a.source_id[i], b.source_id[i]) << "seed " << i;
    }
}

/**
 * Checks that every rank holds the same seeds as rank 0. Collective: rank 0 broadcasts its entry
 * count and then its entries, and every rank makes both broadcasts unconditionally (an early
 * return on a size mismatch would leave the other ranks waiting in the second broadcast); the
 * comparison happens afterwards.
 */
void ExpectSameOnAllRanks(const ms::SeedSet& s) {
    std::vector<double> buf;
    for (std::size_t i = 0; i < s.Size(); i++) {
        buf.push_back(s.x[i]);
        buf.push_back(s.y[i]);
        buf.push_back(s.z[i]);
        buf.push_back(static_cast<double>(s.source_id[i]));
    }
    long long n0 = static_cast<long long>(buf.size());
    MPI_Bcast(&n0, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<double> root(static_cast<std::size_t>(n0));
    if (n0 == static_cast<long long>(buf.size())) {
        root = buf; // rank 0 sends its own entries; others overwrite them
    }
    MPI_Bcast(root.data(), static_cast<int>(n0), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    EXPECT_EQ(static_cast<long long>(buf.size()), n0);
    const std::size_t common = std::min(buf.size(), root.size());
    for (std::size_t i = 0; i < common; i++) {
        EXPECT_EQ(buf[i], root[i]);
    }
}

/** Default parameters: N random seeds by rejection sampling. */
ms::SeedingParameters RandomParams(std::int64_t n, std::uint64_t rng_seed) {
    ms::SeedingParameters p;
    p.num_seeds = n;
    p.rng_seed = rng_seed;
    return p;
}

} // namespace

// Rank-count independence of the default path (rejection + RSA) on the non-convex L-shape, for
// tets (closed-form point-in-tet test) and hexes (Newton inverse map). The distributed run must
// reproduce the single-rank run bitwise, positions and source ids, on every rank.
TEST(Seeding, RankIndependentRejection) {
    for (const bool tets : {true, false}) {
        mfem::Mesh serial = LShape(4, tets);
        ms::SeedingParameters params = RandomParams(150, 11);
        params.min_distance_factor = 0.5;

        const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
        auto pmesh = Distribute(serial);
        ms::SeedingReport report;
        const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);

        EXPECT_EQ(static_cast<std::int64_t>(seeds.Size()), params.num_seeds);
        EXPECT_FALSE(report.used_fallback);
        ExpectSameSeeds(seeds, reference);
        ExpectSameOnAllRanks(seeds);
        // Seed order is increasing candidate index (the source_id contract)
        for (std::size_t i = 1; i < seeds.Size(); i++) {
            EXPECT_LT(seeds.source_id[i - 1], seeds.source_id[i]);
        }
    }
}

// On a box mesh without RSA every candidate is inside the material, so the seeds are exactly
// candidates 0..N-1 and source_id = 0..N-1.
TEST(Seeding, BoxMeshAcceptsEveryCandidate) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, RandomParams(64, 3));
    ASSERT_EQ(seeds.Size(), 64u);
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        EXPECT_EQ(seeds.source_id[i], static_cast<std::int64_t>(i));
    }
}

// Uniformity on a non-convex domain: 3000 seeds in the L-shape (volume 3), counted in 24 equal
// cubes of edge 1/2 (8 per unit cube of the L). For uniform points each count is Binomial with
// mean 125, and the chi-square statistic sum (n_b - 125)^2 / 125 follows a chi-square law with 23
// degrees of freedom, whose 99.9% quantile is 49.73. The RNG key is fixed, so the outcome is
// deterministic; a biased sampler (e.g. one that missed elements near rank boundaries or favored
// the notch side) gives values far above that. Also every seed must lie inside the L.
TEST(Seeding, ChiSquareUniformityLShape) {
    mfem::Mesh serial = LShape(4, true);
    auto pmesh = Distribute(serial);
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, RandomParams(3000, 2024));
    ASSERT_EQ(seeds.Size(), 3000u);

    std::vector<int> counts(4 * 4 * 2, 0); // half-unit cubes over [0,2]x[0,2]x[0,1]
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        ASSERT_TRUE(InLShape(seeds.x[i], seeds.y[i], seeds.z[i]));
        const int bx = std::min(3, static_cast<int>(seeds.x[i] * 2));
        const int by = std::min(3, static_cast<int>(seeds.y[i] * 2));
        const int bz = std::min(1, static_cast<int>(seeds.z[i] * 2));
        counts[bx + 4 * (by + 4 * bz)]++;
    }
    double chi2 = 0.0;
    int bins = 0;
    for (int bz = 0; bz < 2; bz++) {
        for (int by = 0; by < 4; by++) {
            for (int bx = 0; bx < 4; bx++) {
                const int c = counts[bx + 4 * (by + 4 * bz)];
                if (bx >= 2 && by >= 2) {
                    EXPECT_EQ(c, 0); // notch
                    continue;
                }
                chi2 += (c - 125.0) * (c - 125.0) / 125.0;
                bins++;
            }
        }
    }
    EXPECT_EQ(bins, 24);
    EXPECT_LT(chi2, 49.73);
}

// Uniform refinement of a straight-sided mesh does not change the material, so the seeds of the
// once-refined distributed mesh must equal those of the unrefined serial mesh, bitwise, for tets
// and hexes (with RSA, so the min-distance logic is exercised too).
TEST(Seeding, SerialRefinementInvariance) {
    for (const bool tets : {true, false}) {
        mfem::Mesh coarse = LShape(2, tets);
        ms::SeedingParameters params = RandomParams(120, 5);
        params.min_distance_factor = 0.4;
        const ms::SeedSet reference = ms::GenerateSeeds(coarse, MPI_COMM_SELF, params);

        mfem::Mesh fine = LShape(2, tets);
        fine.UniformRefinement(); // ref_ser = 1
        auto pmesh = Distribute(fine);
        const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params);
        ExpectSameSeeds(seeds, reference);
    }
}

// The RSA feasibility guard: with min_distance_factor = 2, N (pi/6) d^3 / V = (pi/6) 8 = 4.2 >
// 0.3, so every rank must throw std::runtime_error with a message saying it is infeasible.
// The decision is made from global data, so no rank is left waiting in a collective call: the
// MPI_Allreduce that combines the per-rank results at the end would hang otherwise.
TEST(Seeding, InfeasibleMinDistanceThrowsOnAllRanks) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::TETRAHEDRON);
    auto pmesh = Distribute(serial);
    ms::SeedingParameters params = RandomParams(100, 1);
    params.min_distance_factor = 2.0;
    bool threw = false;
    try {
        ms::GenerateSeeds(*pmesh, params);
    } catch (const std::runtime_error& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("infeasible"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(threw);
    int all_threw = threw ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_threw, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(all_threw, 1);
}

// Lattice counts in the unit cube. BCC with N = 128: a = (2 V / N)^(1/3) = 1/4, i.e. 4 cubic
// cells per edge with 2 points each, 2 * 4^3 = 128 points, all strictly inside thanks to the
// quarter-cell shift (coordinates 1/16 + k/8). FCC with N = 256: a = (4/256)^(1/3) = 1/4 again and
// 4 * 4^3 = 256 points.
TEST(Seeding, LatticeCountsInBox) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);

    ms::SeedingParameters params = RandomParams(128, 0);
    params.placement = ms::SeedPlacement::LATTICE;
    params.lattice_type = "bcc";
    ms::SeedingReport report;
    const ms::SeedSet bcc = ms::GenerateSeeds(*pmesh, params, &report);
    EXPECT_EQ(bcc.Size(), 128u);
    EXPECT_NEAR(report.lattice_parameter, 0.25, 1e-14);
    for (std::size_t i = 0; i < bcc.Size(); i++) {
        // Every coordinate is an odd multiple of 1/16
        for (double c : {bcc.x[i], bcc.y[i], bcc.z[i]}) {
            const double k = c * 16.0;
            EXPECT_NEAR(k, std::round(k), 1e-9);
            EXPECT_EQ(static_cast<long>(std::round(k)) % 2, 1);
        }
    }

    params.num_seeds = 256;
    params.lattice_type = "fcc";
    const ms::SeedSet fcc = ms::GenerateSeeds(*pmesh, params);
    EXPECT_EQ(fcc.Size(), 256u);

    // Lattices clipped to the L-shape (volume 3): cubic with N = 3 * 64 has a = 1/4 and the cells
    // of the notch are cut away, leaving exactly 3 * 4^3 points.
    mfem::Mesh lshape = LShape(2, true);
    auto pl = Distribute(lshape);
    params.num_seeds = 192;
    params.lattice_type = "cubic";
    const ms::SeedSet cubic = ms::GenerateSeeds(*pl, params);
    EXPECT_EQ(cubic.Size(), 192u);
}

// RSA guarantee: no two seeds closer than the minimum distance d (here 0.6 (V/N)^(1/3)), checked
// over all pairs. Also with periodic axes, where the distance is measured to the nearest periodic
// image (seeds near opposite faces of the cube count as neighbors).
TEST(Seeding, RsaMinimumPairDistance) {
    for (const bool periodic : {false, true}) {
        mfem::Mesh serial = periodic ? UnitCube(4, mfem::Element::HEXAHEDRON) : LShape(3, true);
        auto pmesh = Distribute(serial);
        ms::SeedingParameters params = RandomParams(300, 77);
        params.min_distance_factor = 0.6;
        params.periodic = {periodic, periodic, periodic};
        ms::SeedingReport report;
        const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
        ASSERT_EQ(seeds.Size(), 300u);
        ASSERT_GT(report.min_distance, 0.0);
        EXPECT_GT(report.rejected_by_distance, 0);

        // Same distance function as the grid (minimum image when periodic)
        const double hi = periodic ? 1.0 : 2.0;
        const ms::MinDistanceGrid metric({0.0, 0.0, 0.0}, {hi, hi, 1.0}, report.min_distance,
                                         params.periodic);
        double min_pair = 1e300;
        for (std::size_t i = 0; i < seeds.Size(); i++) {
            for (std::size_t j = i + 1; j < seeds.Size(); j++) {
                min_pair = std::min(min_pair,
                                    metric.Distance({seeds.x[i], seeds.y[i], seeds.z[i]},
                                                    {seeds.x[j], seeds.y[j], seeds.z[j]}));
            }
        }
        EXPECT_GE(min_pair, report.min_distance);
    }
}

// Surface clearance c = 0.15 on the L-shape: every seed must be at least c from the outer faces
// of the block (x, 2-x, y, 2-y, z, 1-z) and from the notch [1,2]x[1,2]x[0,1] (Euclidean distance to
// that box, which is the distance to the two reentrant faces). Checked against a rank-independent
// reference as well, since the surface is split over ranks (shared faces must not count as
// surface).
TEST(Seeding, BoundaryClearanceRespected) {
    mfem::Mesh serial = LShape(4, false);
    ms::SeedingParameters params = RandomParams(100, 9);
    params.boundary_clearance = 0.15;
    const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
    auto pmesh = Distribute(serial);
    ms::SeedingReport report;
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
    ExpectSameSeeds(seeds, reference);
    EXPECT_GT(report.rejected_by_clearance, 0);

    const double c = params.boundary_clearance;
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        const double x = seeds.x[i], y = seeds.y[i], z = seeds.z[i];
        const double outer = std::min({x, 2 - x, y, 2 - y, z, 1 - z});
        EXPECT_GE(outer, c);
        const double dx = std::max(0.0, 1.0 - x);
        const double dy = std::max(0.0, 1.0 - y);
        EXPECT_GE(std::sqrt(dx * dx + dy * dy), c);
    }
}

// Element-weighted sampling, requested explicitly and triggered automatically.
//  - explicit: rank-count independent (elements are keyed by their centroid, not their rank-local
//    index) and every point inside the L-shape;
//  - automatic: a user sampling box of edge 100 around the unit cube makes the material fill
//    1e-6 of the box, below the 1e-4 threshold, so the generator switches to element-weighted
//    sampling (and still returns N in-material seeds).
TEST(Seeding, ElementWeightedSampling) {
    mfem::Mesh serial = LShape(3, true);
    ms::SeedingParameters params = RandomParams(200, 31);
    params.placement = ms::SeedPlacement::ELEMENT_WEIGHTED;
    params.min_distance_factor = 0.3;
    const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
    auto pmesh = Distribute(serial);
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params);
    ExpectSameSeeds(seeds, reference);
    ASSERT_EQ(seeds.Size(), 200u);
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        EXPECT_TRUE(InLShape(seeds.x[i], seeds.y[i], seeds.z[i]));
    }

    mfem::Mesh cube = UnitCube(3, mfem::Element::HEXAHEDRON);
    auto pcube = Distribute(cube);
    ms::SeedingParameters sparse = RandomParams(50, 4);
    sparse.use_sampling_box = true;
    sparse.sampling_box_lo = {-50.0, -50.0, -50.0};
    sparse.sampling_box_hi = {50.0, 50.0, 50.0};
    ms::SeedingReport report;
    const ms::SeedSet fallback = ms::GenerateSeeds(*pcube, sparse, &report);
    EXPECT_TRUE(report.used_fallback);
    EXPECT_EQ(report.placement_used, ms::SeedPlacement::ELEMENT_WEIGHTED);
    ASSERT_EQ(fallback.Size(), 50u);
    for (std::size_t i = 0; i < fallback.Size(); i++) {
        for (double v : {fallback.x[i], fallback.y[i], fallback.z[i]}) {
            EXPECT_GE(v, 0.0);
            EXPECT_LE(v, 1.0);
        }
    }
}

// Stall cap: a feasible but hopeless request (cap of 1000 candidates for 500 RSA seeds) ends
// with an error by default, and with allow_fewer_seeds returns the seeds placed so far and says so
// in the report, identically on all ranks.
TEST(Seeding, StallCapReportsAchievedCount) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::TETRAHEDRON);
    auto pmesh = Distribute(serial);
    ms::SeedingParameters params = RandomParams(500, 2);
    params.min_distance_factor = 0.8;
    params.max_candidates = 1000;
    EXPECT_THROW(ms::GenerateSeeds(*pmesh, params), std::runtime_error);

    params.allow_fewer_seeds = true;
    ms::SeedingReport report;
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
    EXPECT_TRUE(report.stalled);
    EXPECT_EQ(report.achieved, static_cast<std::int64_t>(seeds.Size()));
    EXPECT_LT(report.achieved, 500);
    EXPECT_GT(report.achieved, 0);
    ExpectSameOnAllRanks(seeds);
}

// Meshes with high-order Nodes take the general path: element boxes from sampled points (padded)
// and the Newton inverse map for tets as well as hexes. The unit cube with order-2 Nodes has the
// same (straight) geometry, so every seed must lie in [0,1]^3, and the distributed result must
// still match the single-rank one bitwise.
TEST(Seeding, CurvedNodesPath) {
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = UnitCube(3, type);
        serial.SetCurvature(2);
        ms::SeedingParameters params = RandomParams(60, 8);
        params.min_distance_factor = 0.3;
        params.boundary_clearance = 0.02;
        const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
        auto pmesh = Distribute(serial);
        const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params);
        ExpectSameSeeds(seeds, reference);
        ASSERT_EQ(seeds.Size(), 60u);
        for (std::size_t i = 0; i < seeds.Size(); i++) {
            for (double v : {seeds.x[i], seeds.y[i], seeds.z[i]}) {
                EXPECT_GE(v, 0.02);
                EXPECT_LE(v, 0.98);
            }
        }
    }
}

// Volume guard: moving the center vertex of a 2x2x2 tet cube far above the cube turns the tets
// below it inside out (negative Jacobian determinant, negative volume). GenerateSeeds must refuse
// the mesh with the same error on every rank (here through the element-weighted path, which
// needs positive element volumes for its arrival rates and would otherwise never terminate).
TEST(Seeding, NonPositiveVolumeThrowsOnAllRanks) {
    mfem::Mesh serial = UnitCube(2, mfem::Element::TETRAHEDRON);
    bool moved = false;
    for (int v = 0; v < serial.GetNV(); v++) {
        double* x = serial.GetVertex(v);
        if (std::fabs(x[0] - 0.5) + std::fabs(x[1] - 0.5) + std::fabs(x[2] - 0.5) < 1e-12) {
            x[2] = 5.0;
            moved = true;
        }
    }
    ASSERT_TRUE(moved);
    auto pmesh = Distribute(serial);
    ms::SeedingParameters params = RandomParams(20, 1);
    params.placement = ms::SeedPlacement::ELEMENT_WEIGHTED;
    bool threw = false;
    try {
        ms::GenerateSeeds(*pmesh, params);
    } catch (const std::runtime_error& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("volume"), std::string::npos) << e.what();
    }
    int all_threw = threw ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_threw, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(all_threw, 1);
}

namespace {
/** Bulges the top face of the unit cube: z -> z (1 + 0.2 sin(pi x) sin(pi y)). */
void BulgeTop(const mfem::Vector& x, mfem::Vector& y) {
    const double pi = 3.141592653589793;
    y = x;
    y(2) = x(2) * (1.0 + 0.2 * std::sin(pi * x(0)) * std::sin(pi * x(1)));
}
} // namespace

// A genuinely curved mesh: order-2 Nodes with the top face bulged up to z = 1.2 at its center
// (bulge volume 0.2 (2/pi)^2 = 0.081 of a total ~1.081). Candidates must be drawn from a box that
// reaches into the bulge (the padded element boxes), so about 7.5% of the seeds, ~30 of 400, lie
// above z = 1; with a sampling box limited to the vertices this region would be cut off. Every
// seed must lie under the curved top (within the order-2 interpolation error, 0.02 allowed), and
// the distributed result must match the single-rank one bitwise.
TEST(Seeding, CurvedBulgedMesh) {
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = UnitCube(3, type);
        serial.SetCurvature(2);
        serial.Transform(BulgeTop);
        ms::SeedingParameters params = RandomParams(400, 21);
        const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
        auto pmesh = Distribute(serial);
        ms::SeedingReport report;
        const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
        ExpectSameSeeds(seeds, reference);
        ASSERT_EQ(seeds.Size(), 400u);
        EXPECT_GE(report.sampling_hi[2], 1.2);

        const double pi = 3.141592653589793;
        int above = 0;
        for (std::size_t i = 0; i < seeds.Size(); i++) {
            const double x = seeds.x[i], y = seeds.y[i], z = seeds.z[i];
            EXPECT_GE(x, -1e-12);
            EXPECT_LE(x, 1.0 + 1e-12);
            EXPECT_GE(y, -1e-12);
            EXPECT_LE(y, 1.0 + 1e-12);
            EXPECT_GE(z, -1e-12);
            EXPECT_LE(z, 1.0 + 0.2 * std::sin(pi * x) * std::sin(pi * y) + 0.02);
            above += (z > 1.0) ? 1 : 0;
        }
        EXPECT_GT(above, 10);
        EXPECT_LT(above, 60);
    }
}

// HCP lattice structure: in the ideal hexagonal close-packed lattice every point has exactly 12
// nearest neighbors at distance a (6 in its close-packed layer, 3 above, 3 below) and none closer.
// Checked for every seed farther than 1.01 a from the box faces (whose neighbors all lie inside
// the box). The count of points is about N (boundary effects only), and the result is rank
// independent.
TEST(Seeding, HcpLatticeNeighbors) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    ms::SeedingParameters params = RandomParams(500, 0);
    params.placement = ms::SeedPlacement::LATTICE;
    params.lattice_type = "hcp";
    const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
    auto pmesh = Distribute(serial);
    ms::SeedingReport report;
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
    ExpectSameSeeds(seeds, reference);
    const double a = report.lattice_parameter;
    EXPECT_NEAR(double(seeds.Size()), 500.0, 125.0);

    int interior = 0;
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        const double xi[3] = {seeds.x[i], seeds.y[i], seeds.z[i]};
        bool is_interior = true;
        for (double c : xi) {
            is_interior = is_interior && c > 1.01 * a && c < 1.0 - 1.01 * a;
        }
        if (!is_interior) {
            continue;
        }
        interior++;
        int neighbors = 0;
        for (std::size_t j = 0; j < seeds.Size(); j++) {
            if (j == i) {
                continue;
            }
            const double dx = seeds.x[j] - xi[0], dy = seeds.y[j] - xi[1],
                         dz = seeds.z[j] - xi[2];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            EXPECT_GE(dist, a * (1.0 - 1e-9));
            neighbors += (dist < a * (1.0 + 1e-9)) ? 1 : 0;
        }
        EXPECT_EQ(neighbors, 12);
    }
    EXPECT_GT(interior, 50);
}

// Lattice jitter: cubic lattice in the unit cube, N = 64, a = 1/4, jitter 0.5. Each point moves by
// at most jitter * a / 2 = 1/16 per axis from its site (k + 1/4) a, and since 1/16 < a / 2 the
// nearest site is its own. The 1/4-cell shift keeps every jittered point inside the box (sites
// span [1/16, 13/16] before jitter), so all 64 points remain. The offsets are not all zero, and
// the result is rank independent (jitter is keyed by the lattice index, not by rank).
TEST(Seeding, LatticeJitter) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    ms::SeedingParameters params = RandomParams(64, 13);
    params.placement = ms::SeedPlacement::LATTICE;
    params.lattice_type = "cubic";
    params.lattice_jitter = 0.5;
    const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
    auto pmesh = Distribute(serial);
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params);
    ExpectSameSeeds(seeds, reference);
    ASSERT_EQ(seeds.Size(), 64u);
    double largest_offset = 0.0;
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        for (double c : {seeds.x[i], seeds.y[i], seeds.z[i]}) {
            const double site = (std::round(c * 4.0 - 0.25) + 0.25) / 4.0;
            const double offset = std::fabs(c - site);
            EXPECT_LE(offset, 1.0 / 16.0 + 1e-12);
            largest_offset = std::max(largest_offset, offset);
        }
    }
    EXPECT_GT(largest_offset, 0.01);
}

// Lattice with RSA: a strongly jittered cubic lattice (jitter 1, offsets up to a/2 = 1/8, so
// neighbors can come arbitrarily close) with min_distance 0.15. The RSA filter must remove the
// points that come too close, leaving every pair at least 0.15 apart.
TEST(Seeding, LatticeWithMinDistance) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    ms::SeedingParameters params = RandomParams(64, 5);
    params.placement = ms::SeedPlacement::LATTICE;
    params.lattice_jitter = 1.0;
    params.min_distance = 0.15;
    ms::SeedingReport report;
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
    EXPECT_GT(report.rejected_by_distance, 0);
    EXPECT_LE(seeds.Size(), 64u);
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        for (std::size_t j = i + 1; j < seeds.Size(); j++) {
            const double dx = seeds.x[i] - seeds.x[j], dy = seeds.y[i] - seeds.y[j],
                         dz = seeds.z[i] - seeds.z[j];
            EXPECT_GE(std::sqrt(dx * dx + dy * dy + dz * dz), 0.15);
        }
    }
}

// sampling_domain = "bbox": seeds anywhere in the exact bounding box [0,2] x [0,2] x [0,1] of the
// L-shape, including the notch (a quarter of the box, so ~50 of 200 seeds). No inside test is
// made, so every candidate is accepted and source_id = 0..N-1; rank independent.
TEST(Seeding, BboxSamplingDomain) {
    mfem::Mesh serial = LShape(3, true);
    ms::SeedingParameters params = RandomParams(200, 17);
    params.sampling_domain = "bbox";
    const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
    auto pmesh = Distribute(serial);
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params);
    ExpectSameSeeds(seeds, reference);
    ASSERT_EQ(seeds.Size(), 200u);
    int in_notch = 0;
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        EXPECT_EQ(seeds.source_id[i], static_cast<std::int64_t>(i));
        EXPECT_GE(seeds.x[i], 0.0);
        EXPECT_LE(seeds.x[i], 2.0);
        EXPECT_GE(seeds.y[i], 0.0);
        EXPECT_LE(seeds.y[i], 2.0);
        EXPECT_GE(seeds.z[i], 0.0);
        EXPECT_LE(seeds.z[i], 1.0);
        in_notch += (seeds.x[i] > 1.0 && seeds.y[i] > 1.0) ? 1 : 0;
    }
    EXPECT_GT(in_notch, 25);
    EXPECT_LT(in_notch, 80);
}

// User sampling box with plain rejection: the box [0.5,1.5] x [0.5,1.5] x [0,1] straddles the
// notch corner of the L-shape. Seeds must lie in the box AND in the material (so none in the
// notch quarter x > 1, y > 1 of the box), and the result is rank independent.
TEST(Seeding, UserSamplingBoxRejection) {
    mfem::Mesh serial = LShape(4, false);
    ms::SeedingParameters params = RandomParams(150, 23);
    params.use_sampling_box = true;
    params.sampling_box_lo = {0.5, 0.5, 0.0};
    params.sampling_box_hi = {1.5, 1.5, 1.0};
    const ms::SeedSet reference = ms::GenerateSeeds(serial, MPI_COMM_SELF, params);
    auto pmesh = Distribute(serial);
    ms::SeedingReport report;
    const ms::SeedSet seeds = ms::GenerateSeeds(*pmesh, params, &report);
    ExpectSameSeeds(seeds, reference);
    EXPECT_EQ(report.placement_used, ms::SeedPlacement::REJECTION);
    ASSERT_EQ(seeds.Size(), 150u);
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        EXPECT_GE(seeds.x[i], 0.5);
        EXPECT_LE(seeds.x[i], 1.5);
        EXPECT_GE(seeds.y[i], 0.5);
        EXPECT_LE(seeds.y[i], 1.5);
        EXPECT_TRUE(InLShape(seeds.x[i], seeds.y[i], seeds.z[i]));
    }
}

// Rank-local setup failure: only rank 0 passes an invalid (2D) mesh piece, every other rank a
// valid 3D cube. The failure is detected on rank 0 only, inside the local setup, before any
// collective call; GenerateSeeds must still throw the same std::runtime_error (naming rank 0) on
// every rank instead of leaving ranks 1.. waiting in the first collective.
TEST(Seeding, LocalSetupFailureThrowsOnAllRanks) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mfem::Mesh local = (rank == 0)
                           ? mfem::Mesh::MakeCartesian2D(2, 2, mfem::Element::QUADRILATERAL)
                           : UnitCube(2, mfem::Element::HEXAHEDRON);
    std::string message;
    try {
        ms::GenerateSeeds(local, MPI_COMM_WORLD, RandomParams(10, 0));
    } catch (const std::runtime_error& e) {
        message = e.what();
    }
    EXPECT_NE(message.find("mesh setup failed"), std::string::npos) << message;
    EXPECT_NE(message.find("rank 0"), std::string::npos) << message;
    int all_threw = message.empty() ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &all_threw, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(all_threw, 1);
}

// Invalid parameters are rejected before any work (every rank throws std::invalid_argument).
TEST(Seeding, InvalidParametersRejected) {
    ms::SeedingParameters params = RandomParams(0, 0);
    EXPECT_THROW(ms::ValidateSeedingParameters(params), std::invalid_argument);
    params.num_seeds = 10;
    params.sampling_domain = "everywhere";
    EXPECT_THROW(ms::ValidateSeedingParameters(params), std::invalid_argument);
    params.sampling_domain = "material";
    params.placement = ms::SeedPlacement::LATTICE;
    params.lattice_type = "diamond";
    EXPECT_THROW(ms::ValidateSeedingParameters(params), std::invalid_argument);
    params.placement = ms::SeedPlacement::JOHNSON_MEHL;
    EXPECT_THROW(ms::ValidateSeedingParameters(params), std::invalid_argument);
    // Overflow guards: batches above 2^24 and N above 2^31 - 1
    params.placement = ms::SeedPlacement::REJECTION;
    params.max_batch_size = (std::int64_t(1) << 24) + 1;
    EXPECT_THROW(ms::ValidateSeedingParameters(params), std::invalid_argument);
    params.max_batch_size = 1024;
    params.num_seeds = std::int64_t(1) << 31;
    EXPECT_THROW(ms::ValidateSeedingParameters(params), std::invalid_argument);
}

// RoundBoxOutward: [0.013, 0.987]^2 x [0, 0.5] has largest edge 0.974 = 0.974 * 2^0, so
// q = 2^(0 - 1 - 6) = 1/128; 0.013 goes down to 1/128 and 0.987 up to 127/128, while 0 and 0.5 are
// already multiples of q. Rounding the result again changes nothing.
TEST(Seeding, RoundBoxOutward) {
    std::array<double, 3> lo = {0.013, 0.013, 0.0};
    std::array<double, 3> hi = {0.987, 0.987, 0.5};
    ms::RoundBoxOutward(lo, hi);
    EXPECT_EQ(lo[0], 1.0 / 128.0);
    EXPECT_EQ(hi[0], 127.0 / 128.0);
    EXPECT_EQ(lo[2], 0.0);
    EXPECT_EQ(hi[2], 0.5);
    std::array<double, 3> lo2 = lo;
    std::array<double, 3> hi2 = hi;
    ms::RoundBoxOutward(lo2, hi2);
    EXPECT_EQ(lo2, lo);
    EXPECT_EQ(hi2, hi);
}

// mfem::Mpi::Init initializes MPI and finalizes it automatically at exit, as in the other
// microstructure tests. Every rank runs all tests, since GenerateSeeds is collective.
int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
