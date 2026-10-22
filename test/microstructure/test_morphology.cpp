#include "microstructure/labeling/cell_measures.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/morphology/lloyd.hpp"
#include "microstructure/morphology/sample_labels.hpp"
#include "microstructure/morphology/size_targets.hpp"
#include "microstructure/morphology/weight_fit.hpp"
#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/random_distributions.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

/*
 * Tests for size control: cell measures, target volumes, Laguerre weight fitting and Lloyd
 * relaxation. Every test runs on all ranks (the functions are collective).
 *
 * Rank-count independence is checked inside each run: the distributed result over
 * MPI_COMM_WORLD is compared bitwise with a reference that every rank computes alone on the whole
 * serial mesh (MPI_COMM_SELF), which is exactly the np = 1 answer. Running the executable at
 * np = 1..4 (test_morphology.cmake) therefore checks np = 2, 3, 4 against np = 1. This works
 * because every sum is a fixed-point integer sum (labeling/cell_measures.hpp), so the sample
 * order and distribution do not change any bit of the volumes, Phi or the centroids.
 *
 * Collective calls must be reached by every rank, so the tests use EXPECT_* (which continues
 * after a failure) rather than ASSERT_* (which returns early and would leave the other ranks
 * waiting).
 */

namespace ms = exaconstit::microstructure;

namespace {

/** Unit cube [0,1]^3 with n^3 cells (each split into 6 tets for TETRAHEDRON). */
mfem::Mesh UnitCube(int n, mfem::Element::Type type) {
    return mfem::Mesh::MakeCartesian3D(n, n, n, type, 1.0, 1.0, 1.0, false);
}

/**
 * L-shaped hex block with cells of size 1/n: [0, 1+arm] x [0, 1+arm] x [0, 1] minus the notch
 * [1, 1+arm] x [1, 1+arm] x [0, 1]. Two arms of length `arm` meet at the unit corner square
 * [0,1]^2, so the volume is 1 + 2 arm.
 */
mfem::Mesh LShape(int n, int arm) {
    const int nx = (1 + arm) * n;
    const int ny = nx;
    const int nz = n;
    auto vid = [&](int i, int j, int k) { return i + (nx + 1) * (j + (ny + 1) * k); };
    auto kept = [&](int i, int j) { return !(i >= n && j >= n); };
    int num_cells = 0;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            num_cells += kept(i, j) ? nz : 0;
        }
    }
    mfem::Mesh mesh(3, (nx + 1) * (ny + 1) * (nz + 1), num_cells, 0, 3);
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
                const int hex[8] = {vid(i, j, k),         vid(i + 1, j, k),
                                    vid(i + 1, j + 1, k), vid(i, j + 1, k),
                                    vid(i, j, k + 1),     vid(i + 1, j, k + 1),
                                    vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)};
                mesh.AddHex(hex);
            }
        }
    }
    mesh.FinalizeHexMesh(1, 1, true);
    mesh.RemoveUnusedVertices();
    return mesh;
}

/** True if p lies in the closed L-shape of LShape(n, arm) (with a round-off tolerance). */
bool InLShape(double x, double y, double z, int arm) {
    const double tol = 1e-12;
    const double top = 1.0 + arm;
    const bool in_block = x >= -tol && x <= top + tol && y >= -tol && y <= top + tol &&
                          z >= -tol && z <= 1 + tol;
    const bool in_notch = x > 1 + tol && y > 1 + tol;
    return in_block && !in_notch;
}

/** A serial mesh distributed over a communicator (collective to build). */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/**
 * n seeds uniform in the unit cube from the counter-based RNG (key, GENERIC, i): the same seeds
 * on every rank. source_id = i.
 */
ms::SeedSet RandomSeeds(int n, std::uint64_t key) {
    ms::SeedSet s;
    for (int i = 0; i < n; i++) {
        const ms::CounterRng rng(key, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        double u[3];
        ms::UniformUnitCube(rng, 0, u);
        s.x.push_back(u[0]);
        s.y.push_back(u[1]);
        s.z.push_back(u[2]);
        s.source_id.push_back(i);
    }
    return s;
}

/**
 * Checks that every rank holds the same values as rank 0. Collective: both broadcasts are made
 * unconditionally; the comparison happens afterwards.
 */
void ExpectSameOnAllRanks(const std::vector<double>& v) {
    long long n0 = static_cast<long long>(v.size());
    MPI_Bcast(&n0, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<double> root(static_cast<std::size_t>(n0), 0.0);
    if (n0 == static_cast<long long>(v.size())) {
        root = v;
    }
    MPI_Bcast(root.data(), static_cast<int>(n0), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    EXPECT_EQ(static_cast<long long>(v.size()), n0);
    for (std::size_t i = 0; i < std::min(v.size(), root.size()); i++) {
        EXPECT_EQ(v[i], root[i]) << "entry " << i;
    }
}

/** Bitwise equality of two vectors. */
void ExpectBitwiseEqual(const std::vector<double>& a, const std::vector<double>& b) {
    EXPECT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); i++) {
        EXPECT_EQ(a[i], b[i]) << "entry " << i;
    }
}

/** Largest relative volume error max_i |V_i - T_i| / T_i. */
double MaxRelativeError(const std::vector<double>& volume, const std::vector<double>& target) {
    double err = 0.0;
    for (std::size_t i = 0; i < volume.size(); i++) {
        err = std::max(err, std::fabs(volume[i] - target[i]) / target[i]);
    }
    return err;
}

/** True if every entry is >= the next one minus `slack` (a non-increasing sequence). */
bool NonIncreasing(const std::vector<double>& v, double slack) {
    for (std::size_t i = 1; i < v.size(); i++) {
        if (v[i] > v[i - 1] + slack) {
            return false;
        }
    }
    return true;
}

} // namespace

// Two seeds at x = 0.25 and x = 0.75 in the unit cube: the Voronoi bisector is the plane x = 0.5,
// which is a mesh plane of the 4^3 hex mesh, so each cell holds exactly half of the elements,
// with volume 0.5 and centroid (0.25, 0.5, 0.5) resp. (0.75, 0.5, 0.5). The centroids are exact
// up to the fixed-point quantum and the decode round-off (tolerance 1e-12).
TEST(CellMeasures, BisectorHalves) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    ms::SeedSet seeds;
    seeds.x = {0.25, 0.75};
    seeds.y = {0.5, 0.5};
    seeds.z = {0.5, 0.5};
    seeds.source_id = {0, 1};

    ms::WeightedSamples samples;
    ms::MakeElementWeightedSamples(*dist.pmesh, samples);
    mfem::Array<int> labels;
    ms::LabelSamples(samples, seeds, ms::MetricType::VORONOI, ms::MetricSetup{}, 1, labels,
                     nullptr);
    const ms::CellMeasures m = ms::ComputeCellMeasures(MPI_COMM_WORLD, samples, labels, 2);

    EXPECT_NEAR(m.volume[0], 0.5, 1e-12);
    EXPECT_NEAR(m.volume[1], 0.5, 1e-12);
    EXPECT_NEAR(m.cx[0], 0.25, 1e-12);
    EXPECT_NEAR(m.cx[1], 0.75, 1e-12);
    for (int i = 0; i < 2; i++) {
        EXPECT_NEAR(m.cy[i], 0.5, 1e-12);
        EXPECT_NEAR(m.cz[i], 0.5, 1e-12);
        EXPECT_EQ(m.sample_count[i], 32); // half of the 64 elements
    }
    EXPECT_NEAR(m.total_volume, 1.0, 1e-12);
    EXPECT_EQ(m.total_samples, 64);
}

// Cell measures of random Voronoi cells, from element samples and from quadrature samples:
// (a) bitwise equal to the single-rank reference (fixed-point sums do not depend on the
//     partition), (b) equal to a plain double-precision brute-force sum within round-off,
// (c) volumes sum to the unit-cube volume and counts to the number of samples.
TEST(CellMeasures, RankIndependentAndMatchBruteForce) {
    mfem::Mesh serial = UnitCube(6, mfem::Element::TETRAHEDRON);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::SeedSet seeds = RandomSeeds(12, 7u);

    for (int use_quadrature = 0; use_quadrature < 2; use_quadrature++) {
        auto measure = [&](mfem::Mesh& mesh, MPI_Comm comm, ms::WeightedSamples& samples,
                           mfem::Array<int>& labels) {
            if (use_quadrature) {
                ms::MakeQuadratureWeightedSamples(mesh, -1, samples);
            } else {
                ms::MakeElementWeightedSamples(mesh, samples);
            }
            ms::LabelSamples(samples, seeds, ms::MetricType::VORONOI, ms::MetricSetup{}, 1,
                             labels, nullptr);
            return ms::ComputeCellMeasures(comm, samples, labels, 12);
        };
        ms::WeightedSamples ref_samples, samples;
        mfem::Array<int> ref_labels, labels;
        const ms::CellMeasures ref = measure(serial, MPI_COMM_SELF, ref_samples, ref_labels);
        const ms::CellMeasures got = measure(*dist.pmesh, MPI_COMM_WORLD, samples, labels);

        // (a) rank independence
        ExpectBitwiseEqual(got.volume, ref.volume);
        ExpectBitwiseEqual(got.cx, ref.cx);
        ExpectBitwiseEqual(got.cy, ref.cy);
        ExpectBitwiseEqual(got.cz, ref.cz);
        EXPECT_EQ(got.sample_count, ref.sample_count);
        EXPECT_EQ(got.total_volume, ref.total_volume);

        // (b) brute force on the serial samples (host loop in double precision)
        std::vector<double> v(12, 0.0), mx(12, 0.0);
        const double* w = ref_samples.weight.HostRead();
        const double* x = ref_samples.x.HostRead();
        const int* l = ref_labels.HostRead();
        for (int q = 0; q < ref_samples.Size(); q++) {
            v[l[q]] += w[q];
            mx[l[q]] += w[q] * x[q];
        }
        for (int i = 0; i < 12; i++) {
            EXPECT_NEAR(ref.volume[i], v[i], 1e-13);
            if (v[i] > 0) {
                EXPECT_NEAR(ref.cx[i], mx[i] / v[i], 1e-11);
            }
        }

        // (c) totals
        EXPECT_NEAR(got.total_volume, 1.0, 1e-12);
        EXPECT_EQ(got.total_samples, static_cast<std::int64_t>(ref_samples.Size()));
        std::int64_t count = 0;
        for (auto c : got.sample_count) {
            count += c;
        }
        EXPECT_EQ(count, got.total_samples);
    }
}

// Integrate(1) is the material volume (1 for the unit cube), and Integrate(x) its first moment
// (1/2), identical on all ranks. A label outside [0, N) on the last rank only must make every
// rank throw (the check is reduced), so no rank is left waiting.
TEST(CellMeasures, IntegrateAndBadLabelThrowsOnAllRanks) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    ms::WeightedSamples samples;
    ms::MakeQuadratureWeightedSamples(*dist.pmesh, -1, samples);
    const ms::CellMeasureAccumulator acc(MPI_COMM_WORLD, samples, 3);

    mfem::Vector ones(samples.Size());
    ones = 1.0;
    const double volume = acc.Integrate(ones);
    EXPECT_NEAR(volume, 1.0, 1e-13);
    mfem::Vector xs(samples.x);
    EXPECT_NEAR(acc.Integrate(xs), 0.5, 1e-13);
    ExpectSameOnAllRanks({volume});

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    mfem::Array<int> labels(samples.Size());
    labels = 0;
    if (rank == nranks - 1 && labels.Size() > 0) {
        labels[0] = 3; // invalid: only seeds 0..2 exist
    }
    bool threw = false;
    try {
        acc.Measure(labels);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

// Guards of the fixed-point sums and of the quadrature samples:
// (a) MFEM v4.10 has positive tetrahedron rules only up to order 20 (higher orders fall back to
//     Grundmann-Moller rules with negative weights), so order 21 on a tet ParMesh must throw on
//     every rank, while hexahedron (Gauss) rules of that order are fine;
// (b) an integrand whose largest term is below 1e-280 cannot be given a fixed-point quantum
//     (2^62 / term overflows), so Integrate must refuse it on every rank instead of returning
//     garbage.
TEST(CellMeasures, QuadratureOrderAndScaleGuards) {
    mfem::Mesh tets = UnitCube(2, mfem::Element::TETRAHEDRON);
    Distributed dist_tet = Distribute(tets, MPI_COMM_WORLD);
    ms::WeightedSamples samples;
    EXPECT_NO_THROW(ms::MakeQuadratureWeightedSamples(*dist_tet.pmesh,
                                                      ms::MAX_POSITIVE_TET_QUAD_ORDER, samples));
    EXPECT_THROW(ms::MakeQuadratureWeightedSamples(*dist_tet.pmesh,
                                                   ms::MAX_POSITIVE_TET_QUAD_ORDER + 1, samples),
                 std::runtime_error);
    mfem::Mesh hexes = UnitCube(2, mfem::Element::HEXAHEDRON);
    Distributed dist_hex = Distribute(hexes, MPI_COMM_WORLD);
    const int hex_order = ms::MAX_POSITIVE_TET_QUAD_ORDER + 1;
    EXPECT_NO_THROW(ms::MakeQuadratureWeightedSamples(*dist_hex.pmesh, hex_order, samples));

    ms::MakeQuadratureWeightedSamples(*dist_hex.pmesh, -1, samples);
    const ms::CellMeasureAccumulator acc(MPI_COMM_WORLD, samples, 1);
    mfem::Vector tiny(samples.Size());
    tiny = 1e-300;
    EXPECT_THROW(acc.Integrate(tiny), std::invalid_argument);
}

// Equal targets are V/N each. Log-normal targets: (a) they sum to the material volume, (b) the
// size of a seed depends only on its source id (dropping other seeds keeps the volume ratios),
// (c) the spread of ln V over many grains is 3 sigma (V ~ d^3 and ln d has deviation sigma),
// (d) radius, initial weight and the largest-first order follow the definitions.
TEST(SizeTargets, EqualLognormalAndDerivedData) {
    std::vector<std::int64_t> ids(5);
    for (int i = 0; i < 5; i++) {
        ids[static_cast<std::size_t>(i)] = 10 + i;
    }
    ms::SizeTargetParameters equal;
    const ms::SizeTargets eq = ms::MakeSizeTargets(ids, 2.0, equal);
    for (double v : eq.volume) {
        EXPECT_DOUBLE_EQ(v, 0.4);
    }

    ms::SizeTargetParameters logn;
    logn.distribution = ms::SizeDistribution::LOGNORMAL;
    logn.sigma = 0.35;
    logn.rng_seed = 42;
    const ms::SizeTargets t = ms::MakeSizeTargets(ids, 2.0, logn);
    double sum = 0.0;
    for (double v : t.volume) {
        sum += v;
    }
    EXPECT_NEAR(sum, 2.0, 1e-14); // (a)

    // (b) seeds 11 and 13 alone keep their volume ratio
    const ms::SizeTargets sub = ms::MakeSizeTargets({11, 13}, 1.0, logn);
    EXPECT_NEAR(sub.volume[0] / sub.volume[1], t.volume[1] / t.volume[3], 1e-12);

    // (c) spread of ln V over 20000 grains: standard deviation 3 sigma within 3 percent
    std::vector<std::int64_t> many(20000);
    for (std::size_t i = 0; i < many.size(); i++) {
        many[i] = static_cast<std::int64_t>(i);
    }
    const ms::SizeTargets big = ms::MakeSizeTargets(many, 1.0, logn);
    double mean = 0.0;
    for (double v : big.volume) {
        mean += std::log(v);
    }
    mean /= static_cast<double>(big.volume.size());
    double var = 0.0;
    for (double v : big.volume) {
        var += (std::log(v) - mean) * (std::log(v) - mean);
    }
    var /= static_cast<double>(big.volume.size() - 1);
    EXPECT_NEAR(std::sqrt(var), 3.0 * logn.sigma, 0.03 * 3.0 * logn.sigma);

    // (d) derived data
    const double pi = 3.14159265358979323846;
    for (std::size_t i = 0; i < t.volume.size(); i++) {
        EXPECT_NEAR(4.0 / 3.0 * pi * std::pow(t.radius[i], 3), t.volume[i], 1e-14);
        EXPECT_DOUBLE_EQ(t.initial_weight[i], t.radius[i] * t.radius[i]);
    }
    for (std::size_t k = 1; k < t.placement_order.size(); k++) {
        EXPECT_GE(t.volume[static_cast<std::size_t>(t.placement_order[k - 1])],
                  t.volume[static_cast<std::size_t>(t.placement_order[k])]);
    }
    // Equal volumes: ties ordered by increasing seed index
    EXPECT_EQ(eq.placement_order, (std::vector<int>{0, 1, 2, 3, 4}));
}

// EstimateSeedCount: with equal diameters d, N = V / (pi d^3 / 6); a unit cube with d = 0.2 has
// N = round(1 / 0.0041888) = 239. For the log-normal with the same mean diameter, E[d^3] grows
// by exp(3 sigma^2) (mu = ln E[d] - sigma^2/2), so N shrinks by that factor. Invalid inputs
// throw.
TEST(SizeTargets, SeedCountAndValidation) {
    ms::SizeTargetParameters p;
    p.mean_diameter = 0.2;
    EXPECT_EQ(ms::EstimateSeedCount(1.0, p), 239);
    p.distribution = ms::SizeDistribution::LOGNORMAL;
    p.sigma = 0.3;
    EXPECT_EQ(ms::EstimateSeedCount(1.0, p), std::llround(238.73 / std::exp(3 * 0.09)));

    ms::SizeTargetParameters bad = p;
    bad.sigma = -1.0;
    EXPECT_THROW(ms::ValidateSizeTargetParameters(bad), std::invalid_argument);
    bad.sigma = ms::MAX_LOGNORMAL_SIGMA * 1.01; // beyond the representable bound
    EXPECT_THROW(ms::ValidateSizeTargetParameters(bad), std::invalid_argument);
    // At the bound, with an extreme mean diameter, every target is still finite and positive
    bad.sigma = ms::MAX_LOGNORMAL_SIGMA;
    bad.mean_diameter = 1e100;
    std::vector<std::int64_t> ids(2000);
    for (std::size_t i = 0; i < ids.size(); i++) {
        ids[i] = static_cast<std::int64_t>(i);
    }
    for (double v : ms::MakeSizeTargets(ids, 1.0, bad).volume) {
        EXPECT_TRUE(v > 0.0 && std::isfinite(v));
    }
    bad = p;
    bad.mean_diameter = 0.0;
    EXPECT_THROW(ms::EstimateSeedCount(1.0, bad), std::invalid_argument);
    EXPECT_THROW(ms::MakeSizeTargets({}, 1.0, p), std::invalid_argument);
    EXPECT_THROW(ms::MakeSizeTargets({1}, -1.0, p), std::invalid_argument);
    EXPECT_THROW(ms::MakeSizeTargetsFromVolumes({1.0, 0.0}, 1.0), std::invalid_argument);
    const ms::SizeTargets from = ms::MakeSizeTargetsFromVolumes({1.0, 3.0}, 2.0);
    EXPECT_DOUBLE_EQ(from.volume[0], 0.5);
    EXPECT_DOUBLE_EQ(from.volume[1], 1.5);
}

/**
 * Runs FitWeights on the distributed mesh and on the serial mesh alone (the np = 1 reference)
 * with quadrature samples of the given order (negative: exact default); returns the distributed
 * result and checks bitwise agreement of the weights, the Phi history and the final volumes.
 */
ms::WeightFitResult FitBothWays(mfem::Mesh& serial, const ms::SeedSet& seeds,
                                const std::vector<double>& target,
                                const std::vector<double>& initial,
                                const ms::WeightFitParameters& params, int quad_order = -1) {
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    ms::WeightedSamples samples, ref_samples;
    ms::MakeQuadratureWeightedSamples(*dist.pmesh, quad_order, samples);
    ms::MakeQuadratureWeightedSamples(serial, quad_order, ref_samples);
    const ms::WeightFitResult ref =
        ms::FitWeights(MPI_COMM_SELF, ref_samples, seeds, target, initial, params);
    ms::WeightFitResult got =
        ms::FitWeights(MPI_COMM_WORLD, samples, seeds, target, initial, params);
    ExpectBitwiseEqual(got.weight, ref.weight);
    ExpectBitwiseEqual(got.report.phi, ref.report.phi);
    ExpectBitwiseEqual(got.measures.volume, ref.measures.volume);
    ExpectSameOnAllRanks(got.weight);
    return got;
}

// Equal target volumes for 8 random seeds on a 12^3 hex cube (8 quadrature points per element,
// so the smallest volume step is 1/13824 of the cube): the fit must reach the 2 % tolerance,
// Phi must never decrease along the accepted steps (Armijo guarantees an increase), and the
// result must not depend on the rank count.
TEST(WeightFit, EqualTargetsMonotoneAndRankIndependent) {
    mfem::Mesh serial = UnitCube(12, mfem::Element::HEXAHEDRON);
    const ms::SeedSet seeds = RandomSeeds(8, 3u);
    const std::vector<double> target(8, 1.0 / 8.0);
    ms::WeightFitParameters params;
    const ms::WeightFitResult r = FitBothWays(serial, seeds, target, {}, params);

    EXPECT_TRUE(r.report.converged) << r.report.stop_reason;
    EXPECT_LE(MaxRelativeError(r.measures.volume, r.target_volume), 0.02);
    EXPECT_EQ(r.report.phi.size(), static_cast<std::size_t>(r.report.iterations + 1));
    for (std::size_t k = 1; k < r.report.phi.size(); k++) {
        EXPECT_GT(r.report.phi[k], r.report.phi[k - 1]) << "step " << k;
    }
    EXPECT_GT(r.report.max_relative_error.front(), 0.02); // Voronoi cells start unequal
}

// Log-normal targets (sigma = 0.4) for 20 random seeds on a 12^3 hex cube, started from the
// size-target weights w = r^2: the fit reaches 2 %, monotone and rank independent. Quadrature
// order 5 (27 points per hex) keeps the stated resolution floor (largest sample / smallest
// target) below the 2 % tolerance, so the tolerance is attainable even for the smallest grain.
TEST(WeightFit, LognormalTargets) {
    mfem::Mesh serial = UnitCube(12, mfem::Element::HEXAHEDRON);
    const ms::SeedSet seeds = RandomSeeds(20, 11u);
    ms::SizeTargetParameters sp;
    sp.distribution = ms::SizeDistribution::LOGNORMAL;
    sp.sigma = 0.4;
    sp.rng_seed = 5;
    const ms::SizeTargets targets = ms::MakeSizeTargets(seeds.source_id, 1.0, sp);
    ms::WeightFitParameters params;
    const ms::WeightFitResult r =
        FitBothWays(serial, seeds, targets.volume, targets.initial_weight, params, 5);

    EXPECT_LT(r.report.resolution_floor, 0.02);
    EXPECT_TRUE(r.report.converged) << r.report.stop_reason;
    EXPECT_LE(MaxRelativeError(r.measures.volume, r.target_volume), 0.02);
    for (std::size_t k = 1; k < r.report.phi.size(); k++) {
        EXPECT_GT(r.report.phi[k], r.report.phi[k - 1]) << "step " << k;
    }
}

// A seed whose weight starts far below its neighbors' owns no part of the material (an empty
// Laguerre cell: |x - s|^2 - w is larger than for some other seed everywhere). Its slope
// V^target - 0 > 0 keeps raising its weight, so the cell must reappear and reach its target.
TEST(WeightFit, EmptyCellsRecover) {
    mfem::Mesh serial = UnitCube(10, mfem::Element::TETRAHEDRON);
    const ms::SeedSet seeds = RandomSeeds(10, 21u);
    const std::vector<double> target(10, 0.1);
    std::vector<double> initial(10, 0.0);
    initial[2] = -4.0; // below -(diameter^2 = 3): seed 2 loses everywhere
    initial[7] = -4.0;
    ms::WeightFitParameters params;
    const ms::WeightFitResult r = FitBothWays(serial, seeds, target, initial, params);

    EXPECT_GE(r.report.num_empty.front(), 2);
    EXPECT_EQ(r.report.num_empty.back(), 0);
    EXPECT_TRUE(r.report.converged) << r.report.stop_reason;
    EXPECT_GT(r.measures.volume[2], 0.0);
    EXPECT_GT(r.measures.volume[7], 0.0);
}

// The fit generalizes to every metric of the form f_i = g_i - w_i (the dual and its gradient are
// the same): a Quadric fit with an anisotropic global tensor diag(1, 1, 1/4) (grains elongated
// along z) and an Additive (Apollonius) fit, both with equal targets for 8 seeds, must reach the
// 2 % tolerance with Phi increasing and rank-independent results. They also exercise the
// per-metric first-step scale (InitialInverseStiffness), whose units differ from Laguerre's.
TEST(WeightFit, QuadricAndAdditiveMetrics) {
    mfem::Mesh serial = UnitCube(10, mfem::Element::HEXAHEDRON);
    const ms::SeedSet seeds = RandomSeeds(8, 31u);
    const std::vector<double> target(8, 1.0 / 8.0);
    for (const ms::MetricType metric : {ms::MetricType::QUADRIC, ms::MetricType::ADDITIVE}) {
        ms::WeightFitParameters params;
        params.metric = metric;
        params.setup.global_tensor = {1.0, 1.0, 0.25, 0.0, 0.0, 0.0};
        const ms::WeightFitResult r = FitBothWays(serial, seeds, target, {}, params);
        EXPECT_TRUE(r.report.converged) << r.report.stop_reason;
        EXPECT_LE(MaxRelativeError(r.measures.volume, r.target_volume), 0.02);
        for (std::size_t k = 1; k < r.report.phi.size(); k++) {
            EXPECT_GT(r.report.phi[k], r.report.phi[k - 1]) << "step " << k;
        }
    }
    // First-step scales: Laguerre 0.3 / L, Additive 0.3 / L^2, Quadric 0.3 mu / L with
    // mu = trace(A) / 3; for a mean cell volume of 1/8, L = 1/2.
    ms::MetricSetup setup;
    setup.global_tensor = {1.0, 1.0, 0.25, 0.0, 0.0, 0.0};
    EXPECT_NEAR(ms::InitialInverseStiffness(ms::MetricType::LAGUERRE, 0.125, setup, {}), 0.6,
                1e-14);
    EXPECT_NEAR(ms::InitialInverseStiffness(ms::MetricType::ADDITIVE, 0.125, setup, {}), 1.2,
                1e-14);
    EXPECT_NEAR(ms::InitialInverseStiffness(ms::MetricType::QUADRIC, 0.125, setup, {}), 0.45,
                1e-14);
}

// Phi reported by the fit equals a brute-force evaluation of its definition,
// sum_i w_i V_i^target + sum_q W_q min_i (|x_q - s_i|^2 - w_i), computed here with a plain loop
// over all seeds for every quadrature sample of the serial mesh. The fixed-point sums agree with
// the double-precision loop to round-off (relative 1e-12).
TEST(WeightFit, PhiMatchesBruteForce) {
    mfem::Mesh serial = UnitCube(6, mfem::Element::TETRAHEDRON);
    const ms::SeedSet seeds = RandomSeeds(6, 41u);
    const std::vector<double> target = {0.1, 0.15, 0.2, 0.15, 0.25, 0.15};
    ms::WeightedSamples samples;
    ms::MakeQuadratureWeightedSamples(serial, -1, samples);
    ms::WeightFitParameters params;
    params.max_iterations = 3; // a few steps away from the start, so w is not trivial
    const ms::WeightFitResult r =
        ms::FitWeights(MPI_COMM_SELF, samples, seeds, target, {}, params);

    double phi = 0.0;
    for (std::size_t i = 0; i < seeds.Size(); i++) {
        phi += r.weight[i] * r.target_volume[i];
    }
    const double* x = samples.x.HostRead();
    const double* y = samples.y.HostRead();
    const double* z = samples.z.HostRead();
    const double* w = samples.weight.HostRead();
    for (int q = 0; q < samples.Size(); q++) {
        double min_f = 1e300;
        for (std::size_t i = 0; i < seeds.Size(); i++) {
            const double dx = x[q] - seeds.x[i];
            const double dy = y[q] - seeds.y[i];
            const double dz = z[q] - seeds.z[i];
            min_f = std::min(min_f, dx * dx + dy * dy + dz * dz - r.weight[i]);
        }
        phi += w[q] * min_f;
    }
    EXPECT_NEAR(r.report.phi.back(), phi, 1e-12 * std::fabs(phi));
}

// Resolution floor: with element samples on a 4^3 hex mesh every cell volume is a multiple of
// 1/64, so targets 0.1, 0.2, 0.3, 0.4 can never be met to 1e-9. Phi is then piecewise linear and
// eventually no step increases it: the fit must stop with "line_search_failed" (not by the
// iteration cap), with Phi still strictly increasing over the accepted steps. Before giving up,
// a failed L-BFGS line search is retried once along the scaled steepest direction, which the
// report counts as a restart (at least one here, because the final failure happens with stored
// curvature pairs). The volumes reached must still be within about one element of the targets.
TEST(WeightFit, StopsAtResolutionFloorAfterSteepestRetry) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::SeedSet seeds = RandomSeeds(4, 51u);
    const std::vector<double> target = {0.1, 0.2, 0.3, 0.4};
    ms::WeightFitParameters params;
    params.tolerance = 1e-9;
    params.max_iterations = 2000;
    ms::WeightedSamples samples, ref_samples;
    ms::MakeElementWeightedSamples(*dist.pmesh, samples);
    ms::MakeElementWeightedSamples(serial, ref_samples);
    const ms::WeightFitResult ref =
        ms::FitWeights(MPI_COMM_SELF, ref_samples, seeds, target, {}, params);
    const ms::WeightFitResult r =
        ms::FitWeights(MPI_COMM_WORLD, samples, seeds, target, {}, params);

    EXPECT_EQ(r.report.stop_reason, "line_search_failed");
    EXPECT_FALSE(r.report.converged);
    EXPECT_LT(r.report.iterations, params.max_iterations);
    EXPECT_GE(r.report.num_restarts, 1);
    for (std::size_t k = 1; k < r.report.phi.size(); k++) {
        EXPECT_GT(r.report.phi[k], r.report.phi[k - 1]) << "step " << k;
    }
    for (std::size_t i = 0; i < 4; i++) {
        EXPECT_LE(std::fabs(r.measures.volume[i] - r.target_volume[i]), 1.0 / 64.0 + 1e-12);
    }
    ExpectBitwiseEqual(r.weight, ref.weight);
    EXPECT_EQ(r.report.num_restarts, ref.report.num_restarts);
}

// Parameter validation of the fit and of Lloyd.
TEST(WeightFit, Validation) {
    ms::WeightFitParameters p;
    p.metric = ms::MetricType::VORONOI;
    EXPECT_THROW(ms::ValidateWeightFitParameters(p), std::invalid_argument);
    p = ms::WeightFitParameters{};
    p.tolerance = 0.0;
    EXPECT_THROW(ms::ValidateWeightFitParameters(p), std::invalid_argument);
    p = ms::WeightFitParameters{};
    p.armijo_c = 1.0;
    EXPECT_THROW(ms::ValidateWeightFitParameters(p), std::invalid_argument);
    ms::LloydParameters l;
    l.metric = ms::MetricType::ADDITIVE;
    EXPECT_THROW(ms::ValidateLloydParameters(l), std::invalid_argument);
    l = ms::LloydParameters{};
    l.iterations = -1;
    EXPECT_THROW(ms::ValidateLloydParameters(l), std::invalid_argument);
}

// Lloyd on 16 random seeds in a unit hex cube with quadrature samples: the CVT energy
// sum_i integral_cell_i |x - s_i|^2 must not increase from step to step (moving a seed to its
// cell centroid minimizes the cell's moment of inertia, and relabeling to the nearest seed can
// only lower it further; the cube is convex, so no clamping happens). The slack covers the
// fixed-point resolution of the energy sums. The seeds must be bitwise identical to the
// single-rank run.
TEST(Lloyd, CvtEnergyDecreasesAndRankIndependent) {
    mfem::Mesh serial = UnitCube(8, mfem::Element::HEXAHEDRON);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::SeedSet seeds = RandomSeeds(16, 9u);
    ms::LloydParameters params;
    params.iterations = 10;

    ms::WeightedSamples samples, ref_samples;
    ms::MakeQuadratureWeightedSamples(*dist.pmesh, -1, samples);
    ms::MakeQuadratureWeightedSamples(serial, -1, ref_samples);
    ms::LloydReport report, ref_report;
    const ms::SeedSet ref =
        ms::RunLloyd(serial, MPI_COMM_SELF, ref_samples, seeds, params, {}, &ref_report);
    const ms::SeedSet got =
        ms::RunLloyd(*dist.pmesh, MPI_COMM_WORLD, samples, seeds, params, {}, &report);

    EXPECT_EQ(report.energy.size(), 11u);
    EXPECT_TRUE(NonIncreasing(report.energy, 1e-12));
    EXPECT_LT(report.energy.back(), 0.9 * report.energy.front());
    for (int c : report.num_clamped) {
        EXPECT_EQ(c, 0);
    }
    ExpectBitwiseEqual(got.x, ref.x);
    ExpectBitwiseEqual(got.y, ref.y);
    ExpectBitwiseEqual(got.z, ref.z);
    ExpectBitwiseEqual(report.energy, ref_report.energy);
    EXPECT_EQ(got.source_id, seeds.source_id);
}

// L-shaped block with two arms of length 2: seed 1 near the inner corner owns both arms, whose
// centroids (2, 0.5) and (0.5, 2) average to (1.25, 1.25), inside the notch. Clamping must move
// that seed back to the closest element centroid of its cell, so every seed stays in the
// material after every step; the result must match the single-rank run bitwise.
TEST(Lloyd, ClampingKeepsSeedsInLShape) {
    const int arm = 2;
    mfem::Mesh serial = LShape(4, arm);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    ms::SeedSet seeds;
    seeds.x = {0.1, 0.9};
    seeds.y = {0.1, 0.9};
    seeds.z = {0.5, 0.5};
    seeds.source_id = {0, 1};

    for (int iterations = 1; iterations <= 4; iterations++) {
        ms::LloydParameters params;
        params.iterations = iterations;
        ms::WeightedSamples samples, ref_samples;
        ms::MakeElementWeightedSamples(*dist.pmesh, samples);
        ms::MakeElementWeightedSamples(serial, ref_samples);
        ms::LloydReport report;
        const ms::SeedSet ref =
            ms::RunLloyd(serial, MPI_COMM_SELF, ref_samples, seeds, params, {}, nullptr);
        const ms::SeedSet got =
            ms::RunLloyd(*dist.pmesh, MPI_COMM_WORLD, samples, seeds, params, {}, &report);
        for (std::size_t i = 0; i < got.Size(); i++) {
            EXPECT_TRUE(InLShape(got.x[i], got.y[i], got.z[i], arm))
                << "seed " << i << " at (" << got.x[i] << ", " << got.y[i] << ") after "
                << iterations << " steps";
        }
        EXPECT_GE(report.num_clamped.front(), 1);
        ExpectBitwiseEqual(got.x, ref.x);
        ExpectBitwiseEqual(got.y, ref.y);
    }

    // Without clamping, the first step really does put seed 1 into the notch
    ms::LloydParameters free_params;
    free_params.iterations = 1;
    free_params.clamp_to_material = false;
    ms::WeightedSamples samples;
    ms::MakeElementWeightedSamples(*dist.pmesh, samples);
    const ms::SeedSet moved =
        ms::RunLloyd(*dist.pmesh, MPI_COMM_WORLD, samples, seeds, free_params, {}, nullptr);
    EXPECT_FALSE(InLShape(moved.x[1], moved.y[1], moved.z[1], arm));
}

// Centroidal power mode on 10 random seeds with log-normal targets: every refit (one before the
// first step and one after every step) must converge, the returned weights must reproduce the
// targets within 2 % for the returned positions, and the result must match the single-rank run.
TEST(Lloyd, CentroidalPowerKeepsTargets) {
    mfem::Mesh serial = UnitCube(10, mfem::Element::HEXAHEDRON);
    Distributed dist = Distribute(serial, MPI_COMM_WORLD);
    const ms::SeedSet seeds = RandomSeeds(10, 17u);
    ms::SizeTargetParameters sp;
    sp.distribution = ms::SizeDistribution::LOGNORMAL;
    sp.sigma = 0.3;
    const ms::SizeTargets targets = ms::MakeSizeTargets(seeds.source_id, 1.0, sp);
    ms::SeedSet start = seeds;
    start.weight = targets.initial_weight;

    ms::LloydParameters params;
    params.iterations = 3;
    params.centroidal_power = true;
    ms::WeightedSamples samples, ref_samples;
    ms::MakeQuadratureWeightedSamples(*dist.pmesh, -1, samples);
    ms::MakeQuadratureWeightedSamples(serial, -1, ref_samples);
    ms::LloydReport report;
    const ms::SeedSet ref = ms::RunLloyd(serial, MPI_COMM_SELF, ref_samples, start, params,
                                         targets.volume, nullptr);
    const ms::SeedSet got = ms::RunLloyd(*dist.pmesh, MPI_COMM_WORLD, samples, start, params,
                                         targets.volume, &report);

    EXPECT_EQ(report.fits.size(), 4u);
    for (const auto& fit : report.fits) {
        EXPECT_TRUE(fit.converged) << fit.stop_reason;
    }
    // Volumes of the returned power cells (positions and weights as returned)
    mfem::Array<int> labels;
    ms::LabelSamples(samples, got, ms::MetricType::LAGUERRE, ms::MetricSetup{}, 8, labels,
                     nullptr);
    const ms::CellMeasures m = ms::ComputeCellMeasures(MPI_COMM_WORLD, samples, labels, 10);
    EXPECT_LE(MaxRelativeError(m.volume, targets.volume), 0.02 + 1e-12);
    ExpectBitwiseEqual(got.x, ref.x);
    ExpectBitwiseEqual(got.weight, ref.weight);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
