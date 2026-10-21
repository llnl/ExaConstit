/*
 * Tests of geometry/metric_functors.hpp (the tessellation metrics f_i(x) = g_i(x) - w_i) and
 * geometry/seed_index.hpp (the exact nearest-seed search built on Axom BVHs).
 *
 * The central check is "exactness": SeedIndex skips boxes of seeds using metric lower bounds, and
 * must still return exactly what a brute-force loop over all seeds returns, including the
 * tie-break by lowest seed index and the second-best seed. The analytic tests then check that
 * the metric functions themselves produce the grain-boundary shapes they are documented to
 * produce (bisector planes, shifted planes, Apollonius spheres, hyperboloids).
 *
 * Serial tests on axom::SEQ_EXEC; the host std::mt19937_64 generator is used only to create
 * test inputs (with fixed seeds, so every run is identical).
 */
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/microstructure_types.hpp"

#include "mfem.hpp"

#include "axom/core/execution/execution_space.hpp"
#include "axom/primal/operators/squared_distance.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

using Exec = axom::SEQ_EXEC;
using Point3 = axom::primal::Point<double, 3>;
using Box3 = axom::primal::BoundingBox<double, 3>;

/** @brief Allocator id for SEQ BVH arrays (host memory). */
int HostAllocator() {
    return axom::execution_space<Exec>::allocatorID();
}

/** @brief Query points stored as three mfem::Vectors (the SeedIndex input format). */
struct Queries {
    mfem::Vector x, y, z;
    void Resize(int n) {
        x.SetSize(n);
        y.SetSize(n);
        z.SetSize(n);
    }
    int Size() const {
        return x.Size();
    }
    void Set(int q, double a, double b, double c) {
        x(q) = a;
        y(q) = b;
        z(q) = c;
    }
};

/**
 * @brief Random seed positions in the unit cube: uniform, or "clustered" (10 tight Gaussian
 * clusters, sigma 0.01), which makes the BVH deep and unbalanced and the boxes overlap.
 */
ms::SeedSet MakeSeeds(int n, bool clustered, std::uint64_t rng_seed) {
    std::mt19937_64 gen(rng_seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 0.01);
    double centers[10][3];
    for (auto& c : centers) {
        for (double& v : c) {
            v = 0.1 + 0.8 * uni(gen);
        }
    }
    ms::SeedSet s;
    for (int i = 0; i < n; ++i) {
        if (clustered) {
            const auto& c = centers[i % 10];
            s.x.push_back(c[0] + gauss(gen));
            s.y.push_back(c[1] + gauss(gen));
            s.z.push_back(c[2] + gauss(gen));
        } else {
            s.x.push_back(uni(gen));
            s.y.push_back(uni(gen));
            s.z.push_back(uni(gen));
        }
        s.source_id.push_back(i);
    }
    return s;
}

/** @brief Log-uniform values in [scale * 1e-6, scale]: a spread of six orders of magnitude. */
std::vector<double> LogUniform(int n, double scale, std::uint64_t rng_seed) {
    std::mt19937_64 gen(rng_seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<double> v(n);
    for (double& w : v) {
        w = scale * std::pow(10.0, -6.0 * uni(gen));
    }
    return v;
}

/** @brief Random symmetric positive definite tensor [xx yy zz yz xz xy] = R diag(e) R^T with
 * eigenvalues e in [emin, emax] and a random rotation R. */
void RandomSpdTensor(std::mt19937_64& gen, double emin, double emax, double t[6]) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    // Random rotation from a normalized random quaternion
    double q[4];
    double norm = 0.0;
    for (double& c : q) {
        c = gauss(gen);
        norm += c * c;
    }
    norm = std::sqrt(norm);
    for (double& c : q) {
        c /= norm;
    }
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double r[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    double e[3];
    for (double& v : e) {
        v = emin + (emax - emin) * uni(gen);
    }
    double a[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            a[i][j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                a[i][j] += r[i][k] * e[k] * r[j][k];
            }
        }
    }
    t[0] = a[0][0];
    t[1] = a[1][1];
    t[2] = a[2][2];
    t[3] = a[1][2];
    t[4] = a[0][2];
    t[5] = a[0][1];
}

/**
 * @brief Brute-force reference: loop over all seeds, keep the best two (f, index) pairs, ordered
 * by f and then by index (the documented tie-break). Same conventions as SeedIndex::FindBest
 * with exact_second = true (id2 = -1, f2 = DBL_MAX when there is only one seed).
 */
template <typename Metric>
ms::BestPair BruteForce(const Metric& metric, int n, double x, double y, double z) {
    double f1 = DBL_MAX, f2 = DBL_MAX;
    int id1 = -1, id2 = -1;
    for (int i = 0; i < n; ++i) {
        const double f = metric.Eval(i, x, y, z);
        if (id1 < 0 || f < f1) {
            f2 = f1;
            id2 = id1;
            f1 = f;
            id1 = i;
        } else if (id2 < 0 || f < f2) {
            // seeds are visited in ascending index, so an equal f never displaces an earlier one
            f2 = f;
            id2 = i;
        }
    }
    return ms::BestPair{id1, id2, f1, f2};
}

/**
 * @brief Query set used by the exactness tests: random points in a box 10% larger than the
 * unit cube (so some lie outside the seed cloud), the first seeds themselves (distance zero) and
 * midpoints of random seed pairs (exact or near ties between two seeds).
 */
Queries MakeQueries(const ms::SeedSet& s, int num_random, std::uint64_t rng_seed) {
    std::mt19937_64 gen(rng_seed);
    std::uniform_real_distribution<double> uni(-0.1, 1.1);
    const int n = static_cast<int>(s.Size());
    const int num_at_seeds = std::min(n, 100);
    const int num_mid = (n > 1) ? 100 : 0;
    Queries q;
    q.Resize(num_random + num_at_seeds + num_mid);
    int k = 0;
    for (int i = 0; i < num_random; ++i) {
        const double a = uni(gen), b = uni(gen), c = uni(gen);
        q.Set(k++, a, b, c);
    }
    for (int i = 0; i < num_at_seeds; ++i) {
        q.Set(k++, s.x[i], s.y[i], s.z[i]);
    }
    std::uniform_int_distribution<int> pick(0, std::max(n - 1, 0));
    for (int i = 0; i < num_mid; ++i) {
        const int a = pick(gen), b = pick(gen);
        q.Set(k++, 0.5 * (s.x[a] + s.x[b]), 0.5 * (s.y[a] + s.y[b]), 0.5 * (s.z[a] + s.z[b]));
    }
    return q;
}

/**
 * @brief Builds a SeedIndex<SEQ, Metric>, queries `q` with and without the exact second best,
 * and expects exact agreement (identical indices and bitwise identical f) with BruteForce.
 * Brute force uses the index's own metric functor, so both sides evaluate exactly the same
 * floating-point expressions; any difference is a pruning error.
 */
template <typename Metric>
void ExpectMatchesBruteForce(const ms::SeedSet& s, const ms::MetricSetup& setup,
                             const Queries& q, const std::string& label, int num_bins = 8) {
    ms::SeedIndex<Exec, Metric> index;
    index.Build(s, setup, num_bins, HostAllocator());
    const int n = static_cast<int>(s.Size());

    mfem::Array<ms::BestPair> with_second, without_second;
    index.FindBest(q.x, q.y, q.z, with_second, true);
    index.FindBest(q.x, q.y, q.z, without_second, false);
    const ms::BestPair* ws = with_second.HostRead();
    const ms::BestPair* wo = without_second.HostRead();

    int failures = 0;
    for (int k = 0; k < q.Size() && failures < 10; ++k) {
        const ms::BestPair ref = BruteForce(index.GetMetric(), n, q.x(k), q.y(k), q.z(k));
        const bool ok = ws[k].id1 == ref.id1 && ws[k].f1 == ref.f1 && ws[k].id2 == ref.id2 &&
                        ws[k].f2 == ref.f2 && wo[k].id1 == ref.id1 && wo[k].f1 == ref.f1 &&
                        wo[k].id2 == -1;
        if (!ok) {
            ++failures;
            ADD_FAILURE() << label << ": query " << k << " (" << q.x(k) << ", " << q.y(k) << ", "
                          << q.z(k) << ") index (" << ws[k].id1 << ", " << ws[k].f1 << "; "
                          << ws[k].id2 << ", " << ws[k].f2 << ") first-only (" << wo[k].id1
                          << ", " << wo[k].f1 << ") brute force (" << ref.id1 << ", " << ref.f1
                          << "; " << ref.id2 << ", " << ref.f2 << ")";
        }
    }
}

/** @brief Metric functor over host arrays (for tests of the functors themselves). */
template <typename Metric>
Metric HostMetric(const std::vector<double>& x, const std::vector<double>& y,
                  const std::vector<double>& z, const std::vector<double>& w,
                  const std::vector<double>& factor, int factor_stride) {
    ms::SeedDeviceView v;
    v.x = x.data();
    v.y = y.data();
    v.z = z.data();
    v.weight = w.data();
    v.factor = factor.data();
    v.factor_stride = factor_stride;
    return Metric{v};
}

} // namespace

// -------------------------------------------------------------------------------------------------
// Exactness against brute force
// -------------------------------------------------------------------------------------------------

// All four metrics, uniform and clustered seeds, N = 1, 2, 1000, 20000: the index must return
// exactly the brute-force (id, f) pairs. Weights span six orders of magnitude (Laguerre w in
// [1e-6, 1] h^2, Additive a in [1e-6, 1] h with h the mean seed spacing), so the parameter bins
// really differ; the Quadric cases cover a global SPD tensor with weights, per-seed
// multiplicative weighting (A_i = I / lambda_i^2 with lambda_i^-2 spanning 1e6), per-seed random
// SPD tensors with weights, and a semidefinite (columnar) global tensor, for which the lower
// bound gives no distance pruning.
TEST(SeedIndex, MatchesBruteForceAllMetrics) {
    const int sizes[] = {1, 2, 1000, 20000};
    for (const int n : sizes) {
        for (const bool clustered : {false, true}) {
            const std::string tag =
                "N=" + std::to_string(n) + (clustered ? " clustered" : " uniform");
            const ms::SeedSet base = MakeSeeds(n, clustered, 1000 + n + (clustered ? 7 : 0));
            const Queries q = MakeQueries(base, n >= 20000 ? 600 : 1500, 77 + n);
            const double h = std::cbrt(1.0 / n); // mean seed spacing in the unit cube

            ExpectMatchesBruteForce<ms::VoronoiMetric>(base, ms::MetricSetup{}, q,
                                                       "Voronoi " + tag);

            ms::SeedSet lag = base;
            lag.weight = LogUniform(n, h * h, 5 + n);
            ExpectMatchesBruteForce<ms::LaguerreMetric>(lag, ms::MetricSetup{}, q,
                                                        "Laguerre " + tag);

            ms::SeedSet add = base;
            add.weight = LogUniform(n, h, 9 + n);
            ExpectMatchesBruteForce<ms::AdditiveMetric>(add, ms::MetricSetup{}, q,
                                                        "Additive " + tag);

            // Quadric, global SPD tensor (eigenvalues 0.2 .. 3) plus Laguerre-like weights
            std::mt19937_64 gen(31 + n);
            ms::MetricSetup global;
            RandomSpdTensor(gen, 0.2, 3.0, global.global_tensor.data());
            ExpectMatchesBruteForce<ms::QuadricMetric>(lag, global, q, "Quadric global " + tag);

            // Quadric, multiplicative weighting: A_i = I / lambda_i^2, w = 0
            ms::SeedSet mult = base;
            const std::vector<double> inv_lambda_sq = LogUniform(n, 1.0, 13 + n);
            mult.shape.resize(6 * n, 0.0);
            for (int i = 0; i < n; ++i) {
                mult.shape[6 * i + 0] = inv_lambda_sq[i];
                mult.shape[6 * i + 1] = inv_lambda_sq[i];
                mult.shape[6 * i + 2] = inv_lambda_sq[i];
            }
            ms::MetricSetup per_seed;
            per_seed.per_seed_shape = true;
            ExpectMatchesBruteForce<ms::QuadricMetric>(mult, per_seed, q,
                                                       "Quadric multiplicative " + tag);

            // Quadric, per-seed random SPD tensors and weights
            ms::SeedSet shaped = lag;
            shaped.shape.resize(6 * n);
            for (int i = 0; i < n; ++i) {
                RandomSpdTensor(gen, 0.1, 2.0, &shaped.shape[6 * i]);
            }
            ExpectMatchesBruteForce<ms::QuadricMetric>(shaped, per_seed, q,
                                                       "Quadric per-seed SPD " + tag);

            // Quadric, semidefinite global tensor A = I - e_z e_z^T (columnar grains along z)
            if (n <= 1000) {
                ms::MetricSetup columnar;
                columnar.global_tensor = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
                ExpectMatchesBruteForce<ms::QuadricMetric>(base, columnar, q,
                                                           "Quadric columnar " + tag);
            }
        }
    }
}

// Exact ties across bins: seeds on an integer lattice with Laguerre weights from {0, 0.25, 1}
// (so the three weight values land in different bins) and queries on the half-integer lattice
// and on the seeds. All distances and weights are exact binary fractions, so many queries are
// exactly equidistant (in f) from several seeds in different bins; the lowest index must win,
// which only works if bins and boxes whose bound *equals* the running best are still visited.
TEST(SeedIndex, ExactTiesAcrossBins) {
    ms::SeedSet s;
    const double weights[3] = {0.0, 0.25, 1.0};
    int id = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                s.x.push_back(i);
                s.y.push_back(j);
                s.z.push_back(k);
                s.weight.push_back(weights[(i * 7 + j * 3 + k) % 3]);
                s.source_id.push_back(id++);
            }
        }
    }
    Queries q;
    q.Resize(15 * 15 * 15);
    int k = 0;
    for (int a = 0; a < 15; ++a) {
        for (int b = 0; b < 15; ++b) {
            for (int c = 0; c < 15; ++c) {
                q.Set(k++, 0.5 * a, 0.5 * b, 0.5 * c);
            }
        }
    }
    ExpectMatchesBruteForce<ms::LaguerreMetric>(s, ms::MetricSetup{}, q, "Laguerre lattice");
    ExpectMatchesBruteForce<ms::AdditiveMetric>(s, ms::MetricSetup{}, q, "Additive lattice");
    ExpectMatchesBruteForce<ms::VoronoiMetric>(s, ms::MetricSetup{}, q, "Voronoi lattice");
}

// The number of bins is a performance setting only: 1, 3 and 8 bins give identical answers.
TEST(SeedIndex, ResultIndependentOfBinCount) {
    ms::SeedSet s = MakeSeeds(3000, false, 4242);
    s.weight = LogUniform(3000, 0.005, 99);
    const Queries q = MakeQueries(s, 1000, 5);
    for (const int bins : {1, 3, 8}) {
        ExpectMatchesBruteForce<ms::LaguerreMetric>(s, ms::MetricSetup{}, q,
                                                    "bins=" + std::to_string(bins), bins);
    }
}

// Tie-break where the round-off margin of the lower bound does not help: Laguerre seed A at the
// query point with weight 0 and seed B at distance 1 with weight 1, so f_A = 0 - 0 = 0 and
// f_B = 1 - 1 = 0 exactly. The weights differ, so A and B land in different bins; B's bin has
// the smaller root bound (about -3e-14 < 0) and is searched first, giving best = (0, B). A's bin
// then has root bound exactly 0 (d^2 = 0 and all weights 0, so the margin is 0), equal to the
// best: only the `<=` comparisons (bin loop and box predicate) let the search reach A, which must
// win the tie whenever it has the lower index. Checked with A = seed 0 (A must win) and, as the
// mirror case, A = seed 1 (B = seed 0 wins either way), with and without the second best.
TEST(SeedIndex, TieBreakNeedsLessOrEqual) {
    for (const bool a_is_zero : {true, false}) {
        ms::SeedSet s;
        if (a_is_zero) {
            s.x = {0.0, 1.0};
            s.weight = {0.0, 1.0};
        } else {
            s.x = {1.0, 0.0};
            s.weight = {1.0, 0.0};
        }
        s.y = {0.0, 0.0};
        s.z = {0.0, 0.0};
        ms::SeedIndex<Exec, ms::LaguerreMetric> index;
        index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
        ASSERT_EQ(index.GetReport().num_bins, 2);

        Queries q;
        q.Resize(1);
        q.Set(0, 0.0, 0.0, 0.0);
        mfem::Array<ms::BestPair> first_only, with_second;
        index.FindBest(q.x, q.y, q.z, first_only, false);
        index.FindBest(q.x, q.y, q.z, with_second, true);
        EXPECT_EQ(first_only[0].id1, 0) << (a_is_zero ? "A = seed 0" : "mirror");
        EXPECT_EQ(first_only[0].f1, 0.0);
        EXPECT_EQ(with_second[0].id1, 0);
        EXPECT_EQ(with_second[0].id2, 1);
        EXPECT_EQ(with_second[0].f1, 0.0);
        EXPECT_EQ(with_second[0].f2, 0.0);
    }
}

// Build can be called again on the same index: a rebuild with a different seed count and bin
// count must forget the old seeds entirely (results match brute force for the new seeds, and
// the report describes the new build). Also checks that an empty query set is accepted and
// returns an empty result.
TEST(SeedIndex, RebuildAndEmptyQueries) {
    ms::SeedIndex<Exec, ms::LaguerreMetric> index;
    ms::SeedSet big = MakeSeeds(2000, false, 71);
    big.weight = LogUniform(2000, 1e-3, 72);
    index.Build(big, ms::MetricSetup{}, 8, HostAllocator());
    EXPECT_EQ(index.GetReport().num_bins, 8);

    ms::SeedSet small = MakeSeeds(37, true, 73);
    small.weight = LogUniform(37, 1e-2, 74);
    index.Build(small, ms::MetricSetup{}, 3, HostAllocator());
    EXPECT_EQ(index.GetReport().num_seeds, 37);
    EXPECT_EQ(index.GetReport().num_bins, 3);
    EXPECT_EQ(index.GetRepresentative().Size(), 37);

    const Queries q = MakeQueries(small, 500, 75);
    mfem::Array<ms::BestPair> best;
    index.FindBest(q.x, q.y, q.z, best, true);
    for (int k = 0; k < q.Size(); ++k) {
        const ms::BestPair ref = BruteForce(index.GetMetric(), 37, q.x(k), q.y(k), q.z(k));
        EXPECT_EQ(best[k].id1, ref.id1);
        EXPECT_EQ(best[k].f1, ref.f1);
        EXPECT_EQ(best[k].id2, ref.id2);
        EXPECT_EQ(best[k].f2, ref.f2);
    }

    // Zero queries: empty output, no kernel launch problems
    Queries none;
    none.Resize(0);
    index.FindBest(none.x, none.y, none.z, best, true);
    EXPECT_EQ(best.Size(), 0);
    mfem::Array<int> offsets, ids;
    index.FindWithin(none.x, none.y, none.z, 1.0, offsets, ids);
    EXPECT_EQ(offsets.Size(), 1);
    EXPECT_EQ(ids.Size(), 0);
}

// A non-finite query coordinate (NaN, infinity) makes every comparison fail, so no seed can be
// accepted; the documented result is id1 = id2 = -1 (not an internal sentinel value).
TEST(SeedIndex, NonFiniteQueryGivesMinusOne) {
    ms::SeedSet s = MakeSeeds(50, false, 81);
    ms::SeedIndex<Exec, ms::VoronoiMetric> index;
    index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
    Queries q;
    q.Resize(3);
    q.Set(0, std::nan(""), 0.5, 0.5);
    q.Set(1, 0.5, HUGE_VAL, 0.5);
    q.Set(2, 0.5, 0.5, 0.5); // a normal query for comparison
    mfem::Array<ms::BestPair> best;
    index.FindBest(q.x, q.y, q.z, best, true);
    EXPECT_EQ(best[0].id1, -1);
    EXPECT_EQ(best[0].id2, -1);
    EXPECT_EQ(best[1].id1, -1);
    EXPECT_EQ(best[1].id2, -1);
    EXPECT_GE(best[2].id1, 0);
}

// Exactly coincident seeds (duplicates, including triples, with different weights and different
// quadric tensors) are collapsed into one BVH leaf but still evaluated: results, including the
// second best (which for equal weights is the duplicate itself, at equal f), must match brute
// force, and the report and representative map must name the duplicates.
TEST(SeedIndex, CoincidentSeeds) {
    ms::SeedSet s = MakeSeeds(400, false, 555);
    std::mt19937_64 gen(17);
    std::uniform_int_distribution<int> pick(0, 399);
    // 60 copies of earlier seeds (some seeds get two copies -> triples)
    for (int c = 0; c < 60; ++c) {
        const int src = (c < 20) ? c % 10 : pick(gen);
        s.x.push_back(s.x[src]);
        s.y.push_back(s.y[src]);
        s.z.push_back(s.z[src]);
        s.source_id.push_back(400 + c);
    }
    const int n = static_cast<int>(s.Size());
    s.weight = LogUniform(n, 0.01, 3);
    for (int i = 0; i < 5; ++i) {
        s.weight[400 + i] = s.weight[i]; // some duplicates with exactly equal weight
    }
    s.shape.resize(6 * n);
    for (int i = 0; i < n; ++i) {
        RandomSpdTensor(gen, 0.3, 2.0, &s.shape[6 * i]);
    }

    // Queries: random points plus the duplicated positions themselves
    Queries q = MakeQueries(s, 800, 8);
    const int base = q.Size();
    q.x.SetSize(base + 60);
    q.y.SetSize(base + 60);
    q.z.SetSize(base + 60);
    for (int c = 0; c < 60; ++c) {
        q.Set(base + c, s.x[400 + c], s.y[400 + c], s.z[400 + c]);
    }

    ExpectMatchesBruteForce<ms::VoronoiMetric>(s, ms::MetricSetup{}, q, "Voronoi duplicates");
    ExpectMatchesBruteForce<ms::LaguerreMetric>(s, ms::MetricSetup{}, q, "Laguerre duplicates");
    ExpectMatchesBruteForce<ms::AdditiveMetric>(s, ms::MetricSetup{}, q, "Additive duplicates");
    ms::MetricSetup per_seed;
    per_seed.per_seed_shape = true;
    ExpectMatchesBruteForce<ms::QuadricMetric>(s, per_seed, q, "Quadric duplicates");

    // Report: every copy is coincident with a lower index; the representative is the lowest
    // index at that position
    ms::SeedIndex<Exec, ms::VoronoiMetric> index;
    index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
    EXPECT_EQ(index.GetReport().num_seeds, n);
    EXPECT_EQ(index.GetReport().num_coincident, 60);
    EXPECT_EQ(index.GetReport().num_indexed, n - 60);
    EXPECT_EQ(index.GetReport().num_bins, 1); // Voronoi has no bin parameter
    const int* rep = index.GetRepresentative().HostRead();
    for (int c = 0; c < 60; ++c) {
        const int i = 400 + c;
        EXPECT_LT(rep[i], i);
        EXPECT_EQ(s.x[rep[i]], s.x[i]);
        EXPECT_EQ(rep[rep[i]], rep[i]);
    }
}

// Range query: FindWithin returns exactly the seeds with f <= f_max (ascending index), compared
// with a brute-force filter, for a weighted metric with several bins.
TEST(SeedIndex, FindWithinMatchesBruteForce) {
    ms::SeedSet s = MakeSeeds(800, true, 21);
    s.weight = LogUniform(800, 1e-3, 22);
    const Queries q = MakeQueries(s, 300, 23);
    const double f_max = 2.0e-3;

    ms::SeedIndex<Exec, ms::LaguerreMetric> index;
    index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
    EXPECT_GT(index.GetReport().num_bins, 1);
    mfem::Array<int> offsets, ids;
    index.FindWithin(q.x, q.y, q.z, f_max, offsets, ids);
    ASSERT_EQ(offsets.Size(), q.Size() + 1);

    int total = 0;
    for (int k = 0; k < q.Size(); ++k) {
        std::vector<int> ref;
        for (int i = 0; i < 800; ++i) {
            if (index.GetMetric().Eval(i, q.x(k), q.y(k), q.z(k)) <= f_max) {
                ref.push_back(i);
            }
        }
        total += static_cast<int>(ref.size());
        const std::vector<int> got(ids.GetData() + offsets[k], ids.GetData() + offsets[k + 1]);
        EXPECT_EQ(got, ref) << "query " << k;
    }
    EXPECT_GT(total, 0); // the threshold is large enough that the test is not vacuous
}

// Invalid input is rejected with std::invalid_argument instead of producing a broken index.
TEST(SeedIndex, BuildRejectsInvalidInput) {
    ms::SeedIndex<Exec, ms::QuadricMetric> index;
    ms::SeedSet empty;
    EXPECT_THROW(index.Build(empty, ms::MetricSetup{}, 8, HostAllocator()),
                 std::invalid_argument);

    ms::SeedSet s = MakeSeeds(10, false, 1);
    EXPECT_THROW(index.Build(s, ms::MetricSetup{}, 0, HostAllocator()), std::invalid_argument);

    ms::MetricSetup per_seed; // per-seed tensors requested but SeedSet::shape is empty
    per_seed.per_seed_shape = true;
    EXPECT_THROW(index.Build(s, per_seed, 8, HostAllocator()), std::invalid_argument);

    ms::MetricSetup indefinite; // eigenvalues 1, 1, -1
    indefinite.global_tensor = {1.0, 1.0, -1.0, 0.0, 0.0, 0.0};
    EXPECT_THROW(index.Build(s, indefinite, 8, HostAllocator()), std::invalid_argument);

    s.weight = std::vector<double>(5, 0.0); // wrong size
    EXPECT_THROW(index.Build(s, ms::MetricSetup{}, 8, HostAllocator()), std::invalid_argument);
    EXPECT_FALSE(index.IsBuilt());
}

// -------------------------------------------------------------------------------------------------
// Lower bound property
// -------------------------------------------------------------------------------------------------

/**
 * @brief For random boxes and random seeds inside them, with a BinBound summarizing those seeds,
 * expects LowerBound(squared_distance(x, box), bound) <= Eval(i, x) for every seed and many
 * query points (inside, outside, and exactly at seeds placed on box corners, where the bound is
 * mathematically tight). This is the property that makes the pruned search exact.
 */
template <typename Metric>
void ExpectLowerBoundProperty(bool quadric_per_seed, double weight_scale, const char* name) {
    std::mt19937_64 gen(2024);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    int violations = 0;
    for (int trial = 0; trial < 300; ++trial) {
        // Box with random corner and extents (some flat, some degenerate to a point)
        const double lo[3] = {uni(gen), uni(gen), uni(gen)};
        double hi[3];
        for (int d = 0; d < 3; ++d) {
            const double ext = (trial % 10 == 0) ? 0.0 : 0.3 * uni(gen) * (trial % 7 != 0);
            hi[d] = lo[d] + ext;
        }
        const Box3 box(Point3::make_point(lo[0], lo[1], lo[2]),
                       Point3::make_point(hi[0], hi[1], hi[2]));

        // 12 seeds inside the box: the first two on opposite corners
        const int ns = 12;
        std::vector<double> x(ns), y(ns), z(ns), w(ns), factor(9 * ns);
        ms::BinBound bound;
        bound.w_max = -DBL_MAX;
        bound.w_absmax = 0.0;
        bound.mu_min = DBL_MAX;
        bound.mu_max = 0.0;
        for (int i = 0; i < ns; ++i) {
            const double t[3] = {(i == 0) ? 0.0 : (i == 1) ? 1.0 : uni(gen),
                                 (i == 0) ? 0.0 : (i == 1) ? 1.0 : uni(gen),
                                 (i == 0) ? 0.0 : (i == 1) ? 1.0 : uni(gen)};
            x[i] = lo[0] + t[0] * (hi[0] - lo[0]);
            y[i] = lo[1] + t[1] * (hi[1] - lo[1]);
            z[i] = lo[2] + t[2] * (hi[2] - lo[2]);
            w[i] = weight_scale * (2.0 * uni(gen) - 0.5);
            double tensor[6] = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
            if (quadric_per_seed) {
                RandomSpdTensor(gen, (i % 3 == 0) ? 0.0 : 0.05, 4.0, tensor);
            }
            double mu_lo = 1.0, mu_hi = 1.0;
            ms::ComputeQuadricFactor(tensor, &factor[9 * i], mu_lo, mu_hi);
            bound.w_max = std::max(bound.w_max, w[i]);
            bound.w_absmax = std::max(bound.w_absmax, std::fabs(w[i]));
            bound.mu_min = std::min(bound.mu_min, mu_lo);
            bound.mu_max = std::max(bound.mu_max, mu_hi);
        }
        const Metric metric = HostMetric<Metric>(x, y, z, w, factor, 9);

        // Query points: random in a larger region, plus the seeds themselves
        for (int k = 0; k < 40; ++k) {
            double p[3];
            if (k < ns) {
                p[0] = x[k];
                p[1] = y[k];
                p[2] = z[k];
            } else {
                for (double& c : p) {
                    c = -0.5 + 2.0 * uni(gen);
                }
            }
            const double d2 =
                axom::primal::squared_distance(Point3::make_point(p[0], p[1], p[2]), box);
            const double lb = metric.LowerBound(d2, bound);
            for (int i = 0; i < ns; ++i) {
                const double f = metric.Eval(i, p[0], p[1], p[2]);
                if (!(lb <= f) && violations < 5) {
                    ++violations;
                    ADD_FAILURE() << name << ": LowerBound " << lb << " > f " << f
                                  << " (trial " << trial << ", seed " << i << ")";
                }
            }
        }
    }
}

// LowerBound <= f for all metrics, including weights of both signs and (Quadric) per-seed
// tensors, a third of them semidefinite (mu_min = 0).
TEST(MetricFunctors, LowerBoundNeverExceedsF) {
    ExpectLowerBoundProperty<ms::VoronoiMetric>(false, 0.0, "Voronoi");
    ExpectLowerBoundProperty<ms::LaguerreMetric>(false, 0.05, "Laguerre");
    ExpectLowerBoundProperty<ms::QuadricMetric>(true, 0.05, "Quadric");
    ExpectLowerBoundProperty<ms::AdditiveMetric>(false, 0.2, "Additive");
}

// -------------------------------------------------------------------------------------------------
// Analytic grain-boundary shapes
// -------------------------------------------------------------------------------------------------

// Voronoi bisector: seeds at x = 0.25 and x = 0.75 split space at the plane x = 0.5. On the plane
// both f are exactly 1/16 (binary-exact), so the tie goes to the lowest index, whichever seed
// that is; just off the plane the closer seed wins.
TEST(MetricAnalytic, VoronoiBisectorPlane) {
    for (const bool swapped : {false, true}) {
        ms::SeedSet s;
        s.x = swapped ? std::vector<double>{0.75, 0.25} : std::vector<double>{0.25, 0.75};
        s.y = {0.5, 0.5};
        s.z = {0.5, 0.5};
        const int left = swapped ? 1 : 0; // index of the seed at x = 0.25

        Queries q;
        q.Resize(30);
        std::mt19937_64 gen(3);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int k = 0; k < 10; ++k) {
            const double y = uni(gen), z = uni(gen);
            q.Set(3 * k + 0, 0.5, y, z);
            q.Set(3 * k + 1, 0.5 - 1e-9, y, z);
            q.Set(3 * k + 2, 0.5 + 1e-9, y, z);
        }
        ms::SeedIndex<Exec, ms::VoronoiMetric> index;
        index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
        mfem::Array<ms::BestPair> best;
        index.FindBest(q.x, q.y, q.z, best, true);
        for (int k = 0; k < 10; ++k) {
            EXPECT_EQ(best[3 * k].id1, 0);   // tie -> lowest index
            EXPECT_EQ(best[3 * k].id2, 1);
            EXPECT_EQ(best[3 * k].f1, best[3 * k].f2);
            EXPECT_EQ(best[3 * k + 1].id1, left);
            EXPECT_EQ(best[3 * k + 2].id1, 1 - left);
        }
    }
}

// Laguerre shifted plane: for seeds a, b with weights w_a, w_b the boundary is the plane
// n.x = c with n = s_b - s_a and c = (|s_b|^2 - |s_a|^2 + w_a - w_b) / 2 (expand
// |x - s_a|^2 - w_a = |x - s_b|^2 - w_b). Points projected onto that plane have f_a = f_b up to
// round-off; points moved 1e-6 along -n / +n belong to a / b.
TEST(MetricAnalytic, LaguerreShiftedPlane) {
    ms::SeedSet s;
    s.x = {0.2, 0.7};
    s.y = {0.3, 0.6};
    s.z = {0.4, 0.5};
    s.weight = {0.05, -0.02};
    const double n[3] = {s.x[1] - s.x[0], s.y[1] - s.y[0], s.z[1] - s.z[0]};
    const double nn = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    const double sa2 = s.x[0] * s.x[0] + s.y[0] * s.y[0] + s.z[0] * s.z[0];
    const double sb2 = s.x[1] * s.x[1] + s.y[1] * s.y[1] + s.z[1] * s.z[1];
    const double c = 0.5 * (sb2 - sa2 + s.weight[0] - s.weight[1]);

    ms::SeedIndex<Exec, ms::LaguerreMetric> index;
    index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
    const auto& metric = index.GetMetric();

    std::mt19937_64 gen(11);
    std::uniform_real_distribution<double> uni(-0.5, 1.5);
    Queries q;
    q.Resize(40);
    for (int k = 0; k < 20; ++k) {
        // Project a random point onto the plane n.x = c
        double p[3] = {uni(gen), uni(gen), uni(gen)};
        const double t = (n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - c) / nn;
        for (int d = 0; d < 3; ++d) {
            p[d] -= t * n[d];
        }
        EXPECT_NEAR(metric.Eval(0, p[0], p[1], p[2]), metric.Eval(1, p[0], p[1], p[2]), 1e-14);
        const double step = 1e-6 / std::sqrt(nn);
        q.Set(2 * k, p[0] - step * n[0], p[1] - step * n[1], p[2] - step * n[2]);
        q.Set(2 * k + 1, p[0] + step * n[0], p[1] + step * n[1], p[2] + step * n[2]);
    }
    mfem::Array<ms::BestPair> best;
    index.FindBest(q.x, q.y, q.z, best, false);
    for (int k = 0; k < 20; ++k) {
        EXPECT_EQ(best[2 * k].id1, 0);
        EXPECT_EQ(best[2 * k + 1].id1, 1);
    }
}

// Multiplicative weighting (Quadric with A_i = I / lambda_i^2): the boundary
// |x - a| / lambda_a = |x - b| / lambda_b is the Apollonius sphere with k = lambda_a / lambda_b,
// center (a - k^2 b) / (1 - k^2) and radius k |a - b| / |1 - k^2|. With lambda_a < lambda_b the
// slow grain a is enclosed: points at 0.99 r from the center belong to a, at 1.01 r to b, and on
// the sphere f_a = f_b up to round-off.
TEST(MetricAnalytic, MultiplicativeApolloniusSphere) {
    const double la = 0.5, lb = 1.0;
    ms::SeedSet s;
    s.x = {0.3, 0.7};
    s.y = {0.5, 0.5};
    s.z = {0.5, 0.5};
    s.shape = {1 / (la * la), 1 / (la * la), 1 / (la * la), 0, 0, 0,
               1 / (lb * lb), 1 / (lb * lb), 1 / (lb * lb), 0, 0, 0};
    ms::MetricSetup per_seed;
    per_seed.per_seed_shape = true;
    ms::SeedIndex<Exec, ms::QuadricMetric> index;
    index.Build(s, per_seed, 8, HostAllocator());
    const auto& metric = index.GetMetric();

    const double k = la / lb;
    const double k2 = k * k;
    double center[3];
    const double a[3] = {0.3, 0.5, 0.5}, b[3] = {0.7, 0.5, 0.5};
    for (int d = 0; d < 3; ++d) {
        center[d] = (a[d] - k2 * b[d]) / (1.0 - k2);
    }
    const double radius = k * 0.4 / std::fabs(1.0 - k2);

    std::mt19937_64 gen(5);
    std::normal_distribution<double> gauss(0.0, 1.0);
    Queries q;
    q.Resize(60);
    for (int j = 0; j < 30; ++j) {
        double u[3] = {gauss(gen), gauss(gen), gauss(gen)};
        const double un = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        double on[3];
        for (int d = 0; d < 3; ++d) {
            u[d] /= un;
            on[d] = center[d] + radius * u[d];
        }
        EXPECT_NEAR(metric.Eval(0, on[0], on[1], on[2]), metric.Eval(1, on[0], on[1], on[2]),
                    1e-13);
        q.Set(2 * j, center[0] + 0.99 * radius * u[0], center[1] + 0.99 * radius * u[1],
              center[2] + 0.99 * radius * u[2]);
        q.Set(2 * j + 1, center[0] + 1.01 * radius * u[0], center[1] + 1.01 * radius * u[1],
              center[2] + 1.01 * radius * u[2]);
    }
    mfem::Array<ms::BestPair> best;
    index.FindBest(q.x, q.y, q.z, best, false);
    for (int j = 0; j < 30; ++j) {
        EXPECT_EQ(best[2 * j].id1, 0) << "inside the sphere";
        EXPECT_EQ(best[2 * j + 1].id1, 1) << "outside the sphere";
    }
}

// Additive (Johnson-Mehl) boundary: |x - F1| - |x - F2| = a_1 - a_2 = Delta is one sheet of a
// hyperboloid of revolution with foci F1, F2: with half focal distance c and A = Delta / 2,
// points at axial offset A cosh(u) and radial offset B sinh(u), B^2 = c^2 - A^2, from the focal
// midpoint. On it f_1 = f_2; moving 1e-6 along -/+ grad(f_1 - f_2) (computed with Grad) gives
// grain 1 / grain 2. On the axis the boundary is at x = 0.5 + A.
TEST(MetricAnalytic, AdditiveHyperboloid) {
    ms::SeedSet s;
    s.x = {0.3, 0.7};
    s.y = {0.5, 0.5};
    s.z = {0.5, 0.5};
    s.weight = {0.1, 0.0}; // seed 0 starts earlier (head start 0.1)
    ms::SeedIndex<Exec, ms::AdditiveMetric> index;
    index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
    const auto& metric = index.GetMetric();

    const double cf = 0.2;  // half focal distance
    const double A = 0.05;  // Delta / 2
    const double B = std::sqrt(cf * cf - A * A);
    std::mt19937_64 gen(9);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    Queries q;
    q.Resize(2 * 25 + 2);
    for (int j = 0; j < 25; ++j) {
        const double u = 1.5 * uni(gen);
        const double phi = 6.283185307179586 * uni(gen);
        const double p[3] = {0.5 + A * std::cosh(u), 0.5 + B * std::sinh(u) * std::cos(phi),
                             0.5 + B * std::sinh(u) * std::sin(phi)};
        EXPECT_NEAR(metric.Eval(0, p[0], p[1], p[2]), metric.Eval(1, p[0], p[1], p[2]), 1e-14);
        double g0[3], g1[3], g[3];
        metric.Grad(0, p[0], p[1], p[2], g0);
        metric.Grad(1, p[0], p[1], p[2], g1);
        double gn = 0.0;
        for (int d = 0; d < 3; ++d) {
            g[d] = g0[d] - g1[d];
            gn += g[d] * g[d];
        }
        gn = std::sqrt(gn);
        const double step = 1e-6 / gn;
        q.Set(2 * j, p[0] - step * g[0], p[1] - step * g[1], p[2] - step * g[2]);
        q.Set(2 * j + 1, p[0] + step * g[0], p[1] + step * g[1], p[2] + step * g[2]);
    }
    q.Set(50, 0.5 + A - 1e-6, 0.5, 0.5);
    q.Set(51, 0.5 + A + 1e-6, 0.5, 0.5);
    mfem::Array<ms::BestPair> best;
    index.FindBest(q.x, q.y, q.z, best, false);
    for (int j = 0; j < 26; ++j) {
        EXPECT_EQ(best[2 * j].id1, 0);
        EXPECT_EQ(best[2 * j + 1].id1, 1);
    }
}

// A global quadric tensor A = M^T M is Voronoi in the stretched coordinates y = M x:
// (x - s)^T A (x - s) = |M x - M s|^2. Labels from the Quadric index on (seeds, points) must
// equal labels from a Voronoi index on (M seeds, M points); where they differ, the two seeds must
// be tied in f up to round-off (a point that lies on a grain boundary).
TEST(MetricAnalytic, GlobalTensorIsTransformedVoronoi) {
    ms::MetricSetup setup;
    setup.global_tensor = {2.0, 0.5, 1.0, 0.3, -0.2, 0.4};
    double m[9], mu_lo = 0.0, mu_hi = 0.0;
    ms::ComputeQuadricFactor(setup.global_tensor.data(), m, mu_lo, mu_hi);

    const ms::SeedSet s = MakeSeeds(2000, false, 64);
    const Queries q = MakeQueries(s, 2000, 65);
    ms::SeedSet ts = s; // transformed seeds M s
    Queries tq;
    tq.Resize(q.Size());
    for (std::size_t i = 0; i < s.Size(); ++i) {
        ts.x[i] = m[0] * s.x[i] + m[1] * s.y[i] + m[2] * s.z[i];
        ts.y[i] = m[3] * s.x[i] + m[4] * s.y[i] + m[5] * s.z[i];
        ts.z[i] = m[6] * s.x[i] + m[7] * s.y[i] + m[8] * s.z[i];
    }
    for (int k = 0; k < q.Size(); ++k) {
        tq.Set(k, m[0] * q.x(k) + m[1] * q.y(k) + m[2] * q.z(k),
               m[3] * q.x(k) + m[4] * q.y(k) + m[5] * q.z(k),
               m[6] * q.x(k) + m[7] * q.y(k) + m[8] * q.z(k));
    }

    ms::SeedIndex<Exec, ms::QuadricMetric> quadric;
    quadric.Build(s, setup, 8, HostAllocator());
    ms::SeedIndex<Exec, ms::VoronoiMetric> voronoi;
    voronoi.Build(ts, ms::MetricSetup{}, 8, HostAllocator());
    mfem::Array<ms::BestPair> bq, bv;
    quadric.FindBest(q.x, q.y, q.z, bq, false);
    voronoi.FindBest(tq.x, tq.y, tq.z, bv, false);
    int differ = 0;
    for (int k = 0; k < q.Size(); ++k) {
        if (bq[k].id1 != bv[k].id1) {
            ++differ;
            const auto& metric = quadric.GetMetric();
            EXPECT_NEAR(metric.Eval(bq[k].id1, q.x(k), q.y(k), q.z(k)),
                        metric.Eval(bv[k].id1, q.x(k), q.y(k), q.z(k)), 1e-12)
                << "query " << k;
        }
    }
    EXPECT_LE(differ, q.Size() / 20); // only exact/near ties (seed midpoints) may differ
}

// ComputeQuadricFactor: M^T M reproduces A, eigenvalues come out in range, a semidefinite
// tensor (columnar, A = I - e_z e_z^T) has mu_min = 0, and zero or indefinite tensors are
// rejected.
TEST(MetricFunctors, QuadricFactor) {
    const double t[6] = {2.0, 0.5, 1.0, 0.3, -0.2, 0.4};
    double m[9], mu_lo = 0.0, mu_hi = 0.0;
    ms::ComputeQuadricFactor(t, m, mu_lo, mu_hi);
    const int row[6] = {0, 1, 2, 1, 0, 0};
    const int col[6] = {0, 1, 2, 2, 2, 1};
    for (int e = 0; e < 6; ++e) {
        double a = 0.0;
        for (int r = 0; r < 3; ++r) {
            a += m[3 * r + row[e]] * m[3 * r + col[e]];
        }
        EXPECT_NEAR(a, t[e], 1e-14) << "entry " << e;
    }
    EXPECT_GT(mu_lo, 0.0);
    EXPECT_LE(mu_lo, mu_hi);

    const double columnar[6] = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    ms::ComputeQuadricFactor(columnar, m, mu_lo, mu_hi);
    EXPECT_EQ(mu_lo, 0.0);
    EXPECT_NEAR(mu_hi, 1.0, 1e-15);

    const double zero[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    EXPECT_THROW(ms::ComputeQuadricFactor(zero, m, mu_lo, mu_hi), std::invalid_argument);
    const double indefinite[6] = {1.0, -1.0, 1.0, 0.0, 0.0, 0.0};
    EXPECT_THROW(ms::ComputeQuadricFactor(indefinite, m, mu_lo, mu_hi), std::invalid_argument);
}

// DispatchMetric selects the functor type matching the runtime MetricType.
TEST(MetricFunctors, DispatchMetricSelectsType) {
    const ms::MetricType types[4] = {ms::MetricType::VORONOI, ms::MetricType::LAGUERRE,
                                     ms::MetricType::QUADRIC, ms::MetricType::ADDITIVE};
    for (const ms::MetricType t : types) {
        ms::DispatchMetric(t, [&](auto metric) { EXPECT_EQ(decltype(metric)::TYPE, t); });
    }
}

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
