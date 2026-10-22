#include "microstructure/geometry/anisotropy_tensors.hpp"
#include "microstructure/geometry/metric_functors.hpp"
#include "microstructure/geometry/seed_index.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_types.hpp"
#include "microstructure/seeding/johnson_mehl.hpp"
#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/random_distributions.hpp"

#include "mfem.hpp"

#include "axom/core/execution/execution_space.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Tests of Stage 1c morphology control:
 *  - Johnson-Mehl nucleation and growth (seeding/johnson_mehl.hpp): the analytic two-nucleus
 *    boundary, the "already transformed" rule, the stopping rules, rank-count independence and
 *    the classical grain density;
 *  - columnar and lamellar grains from a global semidefinite quadric tensor
 *    (MakeColumnarTensor / MakeLamellarTensor) and the projected-interval lower bound that keeps
 *    the seed search fast for them (QuadricMetric::LowerBound, box form).
 *
 * Johnson-Mehl tests are collective: the distributed result over MPI_COMM_WORLD is compared
 * bitwise with a reference every rank computes alone on the whole serial mesh (MPI_COMM_SELF),
 * which is exactly the np = 1 answer, so running at np = 1..4 checks rank-count independence.
 * The metric and search tests are serial (every rank runs them on its own); std::mt19937_64 with
 * fixed seeds is used only to create test inputs.
 */

namespace ms = exaconstit::microstructure;

namespace {

using Exec = axom::SEQ_EXEC;

/** Allocator id for SEQ BVH arrays (host memory). */
int HostAllocator() {
    return axom::execution_space<Exec>::allocatorID();
}

int Rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/** Unit cube [0,1]^3 with n^3 cells (6 tets per cell for TETRAHEDRON). */
mfem::Mesh UnitCube(int n, mfem::Element::Type type) {
    return mfem::Mesh::MakeCartesian3D(n, n, n, type, 1.0, 1.0, 1.0, false);
}

/**
 * L-shaped block with cells of size 1/n: [0,2] x [0,2] x [0,1] minus the notch [1,2] x [1,2] x
 * [0,1] (volume 3), hexes or the conforming 6-tet Kuhn split of every cell (same construction as
 * test_seeding.cpp). Non-convex, so the in-material test really rejects candidates.
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
                int c[8]; // corner b = bx + 2 by + 4 bz of the cell
                for (int b = 0; b < 8; b++) {
                    c[b] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
                }
                if (!tets) {
                    const int hex[8] = {c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]};
                    mesh.AddHex(hex);
                } else {
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
    const double tol = 1e-9;
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

/** Bitwise equality of everything the Johnson-Mehl generator fills. */
void ExpectSameNuclei(const ms::SeedSet& a, const ms::SeedSet& b) {
    ASSERT_EQ(a.Size(), b.Size());
    ASSERT_EQ(a.nucleation_time.size(), b.nucleation_time.size());
    ASSERT_EQ(a.weight.size(), b.weight.size());
    for (std::size_t i = 0; i < a.Size(); i++) {
        EXPECT_EQ(a.x[i], b.x[i]) << "nucleus " << i;
        EXPECT_EQ(a.y[i], b.y[i]) << "nucleus " << i;
        EXPECT_EQ(a.z[i], b.z[i]) << "nucleus " << i;
        EXPECT_EQ(a.nucleation_time[i], b.nucleation_time[i]) << "nucleus " << i;
        EXPECT_EQ(a.weight[i], b.weight[i]) << "nucleus " << i;
        EXPECT_EQ(a.source_id[i], b.source_id[i]) << "nucleus " << i;
    }
}

/** Johnson-Mehl parameters with rate I, speed 1, run to full transformation. */
ms::JohnsonMehlParameters JmParams(double rate, std::uint64_t rng_seed) {
    ms::JohnsonMehlParameters p;
    p.nucleation_rate = rate;
    p.growth_speed = 1.0;
    p.rng_seed = rng_seed;
    return p;
}

/** First grain arrival time at x: min_j (t_j + |x - s_j| / v), brute force. */
double ArrivalTime(const ms::SeedSet& s, double v, double x, double y, double z) {
    double best = DBL_MAX;
    for (std::size_t j = 0; j < s.Size(); j++) {
        const double d = std::sqrt((x - s.x[j]) * (x - s.x[j]) + (y - s.y[j]) * (y - s.y[j]) +
                                   (z - s.z[j]) * (z - s.z[j]));
        best = std::min(best, s.nucleation_time[j] + d / v);
    }
    return best;
}

/** Host seed view over plain arrays (for calling metric functors directly in tests). */
ms::SeedDeviceView HostView(const std::vector<double>& x, const std::vector<double>& y,
                            const std::vector<double>& z, const std::vector<double>& w,
                            const double* factor) {
    ms::SeedDeviceView v;
    v.x = x.data();
    v.y = y.data();
    v.z = z.data();
    v.weight = w.data();
    v.factor = factor;
    v.factor_stride = 0;
    return v;
}

/** Random unit vector (Gaussian direction). */
std::array<double, 3> RandomUnit(std::mt19937_64& gen) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::array<double, 3> u = {g(gen), g(gen), g(gen)};
    const double n = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    return {u[0] / n, u[1] / n, u[2] / n};
}

/** A x for the 6-entry tensor [Axx, Ayy, Azz, Ayz, Axz, Axy]. */
std::array<double, 3> Apply(const std::array<double, 6>& a, const std::array<double, 3>& x) {
    return {a[0] * x[0] + a[5] * x[1] + a[4] * x[2], a[5] * x[0] + a[1] * x[1] + a[3] * x[2],
            a[4] * x[0] + a[3] * x[1] + a[2] * x[2]};
}

} // namespace

// ================================================================================================
// Columnar / lamellar tensors and the projected-interval bound (serial on every rank)
// ================================================================================================

// MakeColumnarTensor(a) must be I - a a^T for the normalized axis (a is in its null space,
// directions perpendicular to a are unchanged) and MakeLamellarTensor(n) must be n n^T (n is
// kept, perpendicular directions are annihilated). Axis-aligned inputs give the obvious exact
// entries; zero or non-finite directions are rejected.
TEST(ColumnarLamellar, TensorHelpers) {
    const auto col_z = ms::MakeColumnarTensor({0.0, 0.0, 2.0});
    const std::array<double, 6> expect_col = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    const auto lam_z = ms::MakeLamellarTensor({0.0, 0.0, -3.0});
    const std::array<double, 6> expect_lam = {0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 6; k++) {
        EXPECT_EQ(col_z[k], expect_col[k]);
        EXPECT_EQ(lam_z[k], expect_lam[k]);
    }

    std::mt19937_64 gen(7);
    for (int trial = 0; trial < 20; trial++) {
        const auto a = RandomUnit(gen);
        auto u = RandomUnit(gen); // make u perpendicular to a
        const double ua = u[0] * a[0] + u[1] * a[1] + u[2] * a[2];
        for (int c = 0; c < 3; c++) {
            u[c] -= ua * a[c];
        }
        const std::array<double, 3> a_scaled = {5 * a[0], 5 * a[1], 5 * a[2]};
        const auto col = ms::MakeColumnarTensor(a_scaled);
        const auto lam = ms::MakeLamellarTensor(a_scaled);
        const auto col_a = Apply(col, a);
        const auto col_u = Apply(col, u);
        const auto lam_a = Apply(lam, a);
        const auto lam_u = Apply(lam, u);
        for (int c = 0; c < 3; c++) {
            EXPECT_NEAR(col_a[c], 0.0, 1e-14);
            EXPECT_NEAR(col_u[c], u[c], 1e-14);
            EXPECT_NEAR(lam_a[c], a[c], 1e-14);
            EXPECT_NEAR(lam_u[c], 0.0, 1e-14);
        }
        // Both are accepted as positive semidefinite by the factorization
        double factor[9];
        double mu_min = 0.0;
        double mu_max = 0.0;
        EXPECT_NO_THROW(ms::ComputeQuadricFactor(col.data(), factor, mu_min, mu_max));
        EXPECT_NO_THROW(ms::ComputeQuadricFactor(lam.data(), factor, mu_min, mu_max));
    }
    EXPECT_THROW(ms::MakeColumnarTensor({0.0, 0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(ms::MakeLamellarTensor({NAN, 0.0, 1.0}), std::invalid_argument);
}

// The box form of QuadricMetric::LowerBound must never exceed f_i(x) for any seed i inside the
// box (that is what keeps the search exact). Checked for many random global tensors (positive
// definite, columnar, lamellar, rank 2, nearly singular), random boxes (some far from the
// origin, where round-off is largest), seeds at box corners and inside, queries outside, inside
// and exactly at a seed, and weights of both signs. For a zero-size box (one seed) the bound must
// also be tight: equal to f up to the round-off margin, so it costs no pruning.
TEST(ProjectedIntervalBound, NeverExceedsFAndIsTight) {
    std::mt19937_64 gen(2024);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    long checks = 0;
    long positive_bounds = 0;
    for (int trial = 0; trial < 400; trial++) {
        // Global tensor of one of five kinds
        std::array<double, 6> tensor;
        const int kind = trial % 5;
        const auto a = RandomUnit(gen);
        if (kind == 0) {
            tensor = ms::MakeColumnarTensor(a);
        } else if (kind == 1) {
            tensor = ms::MakeLamellarTensor(a);
        } else {
            // sum_k lambda_k u_k u_k^T with random orthonormal u_k and kind-dependent lambda
            const auto b = RandomUnit(gen);
            std::array<double, 3> u0 = a;
            std::array<double, 3> u1 = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                                        a[0] * b[1] - a[1] * b[0]};
            const double n1 = std::sqrt(u1[0] * u1[0] + u1[1] * u1[1] + u1[2] * u1[2]);
            for (double& c : u1) {
                c /= n1;
            }
            const std::array<double, 3> u2 = {u0[1] * u1[2] - u0[2] * u1[1],
                                              u0[2] * u1[0] - u0[0] * u1[2],
                                              u0[0] * u1[1] - u0[1] * u1[0]};
            double lam[3] = {0.2 + 3 * uni(gen), 0.2 + 3 * uni(gen), 0.2 + 3 * uni(gen)};
            if (kind == 3) {
                lam[2] = 0.0; // rank 2
            }
            if (kind == 4) {
                lam[0] = 1e-9; // nearly singular
            }
            const std::array<double, 3>* u[3] = {&u0, &u1, &u2};
            auto entry = [&](int r, int c) {
                double s = 0.0;
                for (int k = 0; k < 3; k++) {
                    s += lam[k] * (*u[k])[r] * (*u[k])[c];
                }
                return s;
            };
            tensor = {entry(0, 0), entry(1, 1), entry(2, 2), entry(1, 2), entry(0, 2),
                      entry(0, 1)};
        }
        double factor[9];
        double mu_min = 0.0;
        double mu_max = 0.0;
        ms::ComputeQuadricFactor(tensor.data(), factor, mu_min, mu_max);

        // Random box (every fourth trial offset by 1000, where absolute round-off is larger)
        const double offset = (trial % 4 == 3) ? 1000.0 : 0.0;
        double lo[3];
        double hi[3];
        for (int d = 0; d < 3; d++) {
            const double c = offset + 4.0 * uni(gen) - 2.0;
            const double half = (trial % 7 == 0) ? 0.0 : 0.5 * uni(gen); // some zero-size boxes
            lo[d] = c - half;
            hi[d] = c + half;
        }

        // Seeds: the 8 corners and 8 random interior points, weights in [-0.3, 0.3]
        std::vector<double> sx, sy, sz, sw;
        for (int b = 0; b < 16; b++) {
            double p[3];
            for (int d = 0; d < 3; d++) {
                p[d] = (b < 8) ? (((b >> d) & 1) ? hi[d] : lo[d])
                               : lo[d] + (hi[d] - lo[d]) * uni(gen);
            }
            sx.push_back(p[0]);
            sy.push_back(p[1]);
            sz.push_back(p[2]);
            sw.push_back((trial % 3 == 0) ? 0.0 : 0.6 * uni(gen) - 0.3);
        }
        ms::BinBound bound;
        bound.w_max = *std::max_element(sw.begin(), sw.end());
        bound.w_absmax = 0.0;
        for (double w : sw) {
            bound.w_absmax = std::max(bound.w_absmax, std::fabs(w));
        }
        bound.mu_min = mu_min;
        bound.mu_max = mu_max;
        const ms::QuadricMetric metric{HostView(sx, sy, sz, sw, factor)};

        // Queries: random near and far points, the box center, and each seed position
        for (int q = 0; q < 40; q++) {
            double p[3];
            for (int d = 0; d < 3; d++) {
                if (q < 16) {
                    p[d] = (d == 0) ? sx[q] : (d == 1 ? sy[q] : sz[q]);
                } else if (q == 16) {
                    p[d] = 0.5 * (lo[d] + hi[d]);
                } else {
                    p[d] = offset + 8.0 * uni(gen) - 4.0;
                }
            }
            double sqdist = 0.0;
            for (int d = 0; d < 3; d++) {
                const double gap = std::max({lo[d] - p[d], 0.0, p[d] - hi[d]});
                sqdist += gap * gap;
            }
            const double lb = metric.LowerBound(p, lo, hi, sqdist, bound);
            positive_bounds += (lb > 0.0) ? 1 : 0;
            for (int i = 0; i < 16; i++) {
                const double f = metric.Eval(i, p[0], p[1], p[2]);
                checks++;
                if (lb > f) {
                    ADD_FAILURE() << "kind " << kind << " trial " << trial << ": bound " << lb
                                  << " > f " << f;
                }
            }
            // Tightness for a zero-size box: lb close to f of the (single) seed position
            if (lo[0] == hi[0] && lo[1] == hi[1] && lo[2] == hi[2] && trial % 3 == 0) {
                const double f = metric.Eval(0, p[0], p[1], p[2]);
                EXPECT_GE(lb, f - 1e-9 * (1.0 + std::fabs(f)) * (1.0 + offset * offset));
            }
        }
    }
    EXPECT_GT(checks, 200000);
    EXPECT_GT(positive_bounds, 1000); // the bound really prunes (is often above 0)
}

// Columnar grains (A = I - a a^T, tilted axis) with 4000 random seeds: SeedIndex with the global
// tensor (projected-interval bound) must give exactly the brute-force best and second-best seed,
// with and without weights, and visit far fewer BVH leaves than the same metric given as
// identical per-seed tensors (which uses only the eigenvalue bound mu_min d^2 - w_max; with
// mu_min = 0 that bound prunes nothing, so every leaf is visited). The leaf counts are printed.
TEST(ColumnarLamellar, ColumnarIndexMatchesBruteForceAndPrunes) {
    const int n = 4000;
    const int nq = 1500;
    std::mt19937_64 gen(11);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::array<double, 6> tensor = ms::MakeColumnarTensor({1.0, 2.0, 3.0});

    for (int weighted = 0; weighted < 2; weighted++) {
        ms::SeedSet seeds;
        for (int i = 0; i < n; i++) {
            seeds.x.push_back(uni(gen));
            seeds.y.push_back(uni(gen));
            seeds.z.push_back(uni(gen));
            seeds.weight.push_back(weighted ? 0.002 * uni(gen) : 0.0);
            for (int k = 0; k < 6; k++) {
                seeds.shape.push_back(tensor[k]);
            }
        }
        mfem::Vector qx(nq), qy(nq), qz(nq);
        for (int q = 0; q < nq; q++) {
            qx(q) = uni(gen);
            qy(q) = uni(gen);
            qz(q) = uni(gen);
        }

        ms::MetricSetup global;
        global.global_tensor = tensor;
        ms::MetricSetup per_seed;
        per_seed.per_seed_shape = true;
        ms::SeedIndex<Exec, ms::QuadricMetric> fast;
        ms::SeedIndex<Exec, ms::QuadricMetric> slow;
        fast.Build(seeds, global, 8, HostAllocator());
        slow.Build(seeds, per_seed, 8, HostAllocator());

        mfem::Array<ms::BestPair> best_fast, best_slow;
        mfem::Array<int> visits_fast, visits_slow;
        fast.FindBest(qx, qy, qz, best_fast, true, &visits_fast);
        slow.FindBest(qx, qy, qz, best_slow, true, &visits_slow);

        // Brute force with the index's own metric functor (host pointers for SEQ)
        const ms::QuadricMetric& metric = fast.GetMetric();
        double sum_fast = 0.0;
        double sum_slow = 0.0;
        for (int q = 0; q < nq; q++) {
            int id1 = -1, id2 = -1;
            double f1 = DBL_MAX, f2 = DBL_MAX;
            for (int i = 0; i < n; i++) {
                const double f = metric.Eval(i, qx(q), qy(q), qz(q));
                if (f < f1 || (f == f1 && i < id1)) {
                    f2 = f1;
                    id2 = id1;
                    f1 = f;
                    id1 = i;
                } else if (f < f2 || (f == f2 && i < id2)) {
                    f2 = f;
                    id2 = i;
                }
            }
            EXPECT_EQ(best_fast[q].id1, id1) << "query " << q;
            EXPECT_EQ(best_fast[q].f1, f1) << "query " << q;
            EXPECT_EQ(best_fast[q].id2, id2) << "query " << q;
            EXPECT_EQ(best_slow[q].id1, id1) << "query " << q;
            EXPECT_EQ(best_slow[q].id2, id2) << "query " << q;
            sum_fast += visits_fast[q];
            sum_slow += visits_slow[q];
            EXPECT_EQ(visits_slow[q], n) << "mu_min = 0 bound should visit every leaf";
        }
        const double mean_fast = sum_fast / nq;
        const double mean_slow = sum_slow / nq;
        if (Rank() == 0) {
            std::cout << "[ columnar " << (weighted ? "weighted" : "unweighted") << " ] " << n
                      << " seeds, mean BVH leaves visited per query: projected-interval bound "
                      << mean_fast << ", mu_min bound " << mean_slow << std::endl;
        }
        EXPECT_LT(mean_fast, 0.1 * mean_slow);
    }
}

// Lamellar grains (A = n n^T): f_i depends only on the height h = n . x, so the grain of x must be
// the seed whose height n . s_i is closest to h (ties: lowest index), i.e. the layers are flat
// slabs bounded by the planes halfway between consecutive seed heights. Checked against that
// independent 1D rule for a tilted normal, skipping points within 1e-9 of a mid-plane, and the
// in-plane invariance is checked exactly for n = z (moving x, y never changes the label).
TEST(ColumnarLamellar, LamellarLayersArePlanar) {
    std::mt19937_64 gen(5);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::array<std::array<double, 3>, 2> normals = {{{0.3, -0.5, 0.8}, {0.0, 0.0, 1.0}}};
    for (const auto& normal_in : normals) {
        const double len = std::sqrt(normal_in[0] * normal_in[0] + normal_in[1] * normal_in[1] +
                                     normal_in[2] * normal_in[2]);
        const std::array<double, 3> nrm = {normal_in[0] / len, normal_in[1] / len,
                                           normal_in[2] / len};
        ms::SeedSet seeds;
        std::vector<double> height;
        for (int i = 0; i < 60; i++) {
            seeds.x.push_back(uni(gen));
            seeds.y.push_back(uni(gen));
            seeds.z.push_back(uni(gen));
            height.push_back(nrm[0] * seeds.x[i] + nrm[1] * seeds.y[i] + nrm[2] * seeds.z[i]);
        }
        ms::MetricSetup setup;
        setup.global_tensor = ms::MakeLamellarTensor(normal_in);
        ms::SeedIndex<Exec, ms::QuadricMetric> index;
        index.Build(seeds, setup, 1, HostAllocator());

        const int nq = 3000;
        mfem::Vector qx(nq), qy(nq), qz(nq);
        for (int q = 0; q < nq; q++) {
            qx(q) = uni(gen);
            qy(q) = uni(gen);
            qz(q) = uni(gen);
        }
        mfem::Array<ms::BestPair> best;
        index.FindBest(qx, qy, qz, best, false);

        int compared = 0;
        for (int q = 0; q < nq; q++) {
            const double h = nrm[0] * qx(q) + nrm[1] * qy(q) + nrm[2] * qz(q);
            // Closest seed height and the margin to the second closest (distance to mid-plane)
            int id = -1;
            double d1 = DBL_MAX, d2 = DBL_MAX;
            for (int i = 0; i < 60; i++) {
                const double d = std::fabs(h - height[i]);
                if (d < d1) {
                    d2 = d1;
                    d1 = d;
                    id = i;
                } else if (d < d2) {
                    d2 = d;
                }
            }
            if (d2 - d1 < 1e-9) {
                continue;
            }
            compared++;
            EXPECT_EQ(best[q].id1, id) << "query " << q;
        }
        EXPECT_GT(compared, nq - 10);

        // Exact in-plane invariance for n = z: same z, new x and y, same label
        if (nrm[2] == 1.0) {
            mfem::Vector sx(nq), sy(nq);
            for (int q = 0; q < nq; q++) {
                sx(q) = uni(gen);
                sy(q) = uni(gen);
            }
            mfem::Array<ms::BestPair> shifted;
            index.FindBest(sx, sy, qz, shifted, false);
            for (int q = 0; q < nq; q++) {
                EXPECT_EQ(shifted[q].id1, best[q].id1) << "query " << q;
            }
        }
    }
}

// ================================================================================================
// Johnson-Mehl nucleation and growth
// ================================================================================================

// Two nuclei born at s_a (time t_a) and s_b (later, t_b) growing at speed v meet where both
// arrive at the same time: |x - s_a| - |x - s_b| = v (t_b - t_a) = 2 alpha. That is one sheet of
// a hyperboloid of revolution with foci s_a, s_b: in coordinates centered between the foci, z
// along s_b - s_a and rho the distance from that axis, z^2 / alpha^2 - rho^2 / (c^2 - alpha^2)
// = 1 with z > 0 (c = half the focal distance); the later grain b owns the convex region inside
// the sheet. Labels from SeedIndex<AdditiveMetric> with the generator's weights a = -v t must
// match this closed-form quadric, for random points and for points placed on the sheet and
// nudged 1e-7 to either side along its normal.
TEST(JohnsonMehl, TwoNucleusBoundaryIsHyperboloid) {
    const double v = 2.0;
    const double t_a = 0.05;
    const double t_b = 0.15;
    const std::array<double, 3> s_a = {0.2, 0.3, 0.4};
    const std::array<double, 3> s_b = {0.8, 0.6, 0.5};
    ms::SeedSet seeds;
    for (const auto* s : {&s_a, &s_b}) {
        seeds.x.push_back((*s)[0]);
        seeds.y.push_back((*s)[1]);
        seeds.z.push_back((*s)[2]);
    }
    seeds.nucleation_time = {t_a, t_b};
    seeds.weight = {-v * t_a, -v * t_b}; // the Johnson-Mehl convention a_i = -v t_i
    ms::SeedIndex<Exec, ms::AdditiveMetric> index;
    index.Build(seeds, ms::MetricSetup{}, 2, HostAllocator());

    // Hyperboloid frame: center, unit axis e (a -> b), two perpendicular unit vectors
    std::array<double, 3> center, e;
    double focal = 0.0;
    for (int d = 0; d < 3; d++) {
        center[d] = 0.5 * (s_a[d] + s_b[d]);
        e[d] = s_b[d] - s_a[d];
        focal += e[d] * e[d];
    }
    focal = std::sqrt(focal);
    for (double& c : e) {
        c /= focal;
    }
    const double c_half = 0.5 * focal;
    const double alpha = 0.5 * v * (t_b - t_a);
    const double beta_sq = c_half * c_half - alpha * alpha;
    ASSERT_GT(beta_sq, 0.0);
    std::array<double, 3> e1 = {-e[1], e[0], 0.0};
    const double n1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1]);
    for (double& c : e1) {
        c /= n1;
    }
    const std::array<double, 3> e2 = {e[1] * e1[2] - e[2] * e1[1], e[2] * e1[0] - e[0] * e1[2],
                                      e[0] * e1[1] - e[1] * e1[0]};

    // Closed-form side test: +1 inside the sheet around b, -1 outside, 0 = too close to decide
    auto side = [&](const std::array<double, 3>& x) {
        double z = 0.0, rho_sq = 0.0, r_sq = 0.0;
        for (int d = 0; d < 3; d++) {
            z += (x[d] - center[d]) * e[d];
            r_sq += (x[d] - center[d]) * (x[d] - center[d]);
        }
        rho_sq = r_sq - z * z;
        const double q = z * z / (alpha * alpha) - rho_sq / beta_sq - 1.0;
        if (std::fabs(q) < 1e-6) {
            return 0;
        }
        return (z > 0.0 && q > 0.0) ? 1 : -1;
    };

    // Random points plus nudged points on the sheet
    std::mt19937_64 gen(3);
    std::uniform_real_distribution<double> uni(-0.5, 1.5);
    std::uniform_real_distribution<double> ang(0.0, 6.283185307179586);
    std::uniform_real_distribution<double> par(0.0, 1.5);
    std::vector<std::array<double, 3>> pts;
    std::vector<int> expected;
    for (int k = 0; k < 4000; k++) {
        const std::array<double, 3> x = {uni(gen), uni(gen), uni(gen)};
        const int s = side(x);
        if (s != 0) {
            pts.push_back(x);
            expected.push_back(s > 0 ? 1 : 0);
        }
    }
    const double beta = std::sqrt(beta_sq);
    for (int k = 0; k < 500; k++) {
        // Point on the sheet: z = alpha cosh(u), rho = beta sinh(u), angle phi
        const double u = par(gen);
        const double phi = ang(gen);
        const double z = alpha * std::cosh(u);
        const double rho = beta * std::sinh(u);
        // Normal of the level set q(z, rho) = z^2/alpha^2 - rho^2/beta^2, pointing to larger q:
        // the gradient (2 z / alpha^2, -2 rho / beta^2), normalized
        double nz = z / (alpha * alpha);
        double nr = -rho / beta_sq;
        const double nn = std::sqrt(nz * nz + nr * nr);
        nz /= nn;
        nr /= nn;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            const double zz = z + sgn * 1e-7 * nz;
            const double rr = rho + sgn * 1e-7 * nr;
            std::array<double, 3> x;
            for (int d = 0; d < 3; d++) {
                x[d] = center[d] + zz * e[d] + rr * (std::cos(phi) * e1[d] + std::sin(phi) * e2[d]);
            }
            pts.push_back(x);
            expected.push_back(sgn > 0 ? 1 : 0); // + along the gradient of q: inside b's sheet
        }
    }
    const int np = static_cast<int>(pts.size());
    mfem::Vector qx(np), qy(np), qz(np);
    for (int k = 0; k < np; k++) {
        qx(k) = pts[k][0];
        qy(k) = pts[k][1];
        qz(k) = pts[k][2];
    }
    mfem::Array<ms::BestPair> best;
    index.FindBest(qx, qy, qz, best, false);
    for (int k = 0; k < np; k++) {
        EXPECT_EQ(best[k].id1, expected[k]) << "point " << k << " (" << pts[k][0] << ", "
                                            << pts[k][1] << ", " << pts[k][2] << ")";
    }
}

// The defining rules of the process, on the non-convex L-shape (tets), run to full
// transformation with every discarded candidate recorded:
//  - every discarded candidate really was covered: the reported nucleus j was born earlier and
//    |c - s_j| < v (t - t_j);
//  - no accepted nucleus was covered by an earlier accepted one (brute force over all pairs);
//  - nuclei are in the material, in birth order, with a_i = -v t_i and increasing source ids;
//  - the completion bound is a true bound: the brute-force arrival time at every mesh vertex,
//    element centroid and random element point is below it, and generation stopped because
//    the material was fully transformed;
//  - stopping early changed nothing: the regenerated candidate stream after the stop has every
//    in-material candidate already covered;
//  - every nucleus wins its own position (f_i(s_i) is minimal), so no Johnson-Mehl grain is
//    empty; this follows from the two coverage rules (see the argument in the comment below).
TEST(JohnsonMehl, CoverageRulesAndCompletion) {
    mfem::Mesh serial = LShape(4, true);
    auto pmesh = Distribute(serial);
    ms::JohnsonMehlParameters params = JmParams(2000.0, 42);
    params.growth_speed = 1.5;
    params.record_rejected = true;
    ms::JohnsonMehlReport report;
    const ms::SeedSet s = ms::GenerateJohnsonMehlSeeds(*pmesh, params, &report);
    const double v = params.growth_speed;

    ASSERT_GT(s.Size(), 10u);
    EXPECT_EQ(report.stop, ms::JohnsonMehlStop::FULLY_TRANSFORMED);
    EXPECT_EQ(report.accepted, static_cast<std::int64_t>(s.Size()));
    EXPECT_GT(report.rejected_covered, 0);
    EXPECT_EQ(report.rejected_by_rate, 0);
    EXPECT_EQ(report.candidates_in_material, report.accepted + report.rejected_covered);
    ASSERT_EQ(report.rejected_time.size(), static_cast<std::size_t>(report.rejected_covered));

    // Accepted nuclei: in material, ordered, consistent weights and ids
    for (std::size_t i = 0; i < s.Size(); i++) {
        EXPECT_TRUE(InLShape(s.x[i], s.y[i], s.z[i])) << "nucleus " << i;
        EXPECT_EQ(s.weight[i], -v * s.nucleation_time[i]);
        if (i > 0) {
            EXPECT_GE(s.nucleation_time[i], s.nucleation_time[i - 1]);
            EXPECT_GT(s.source_id[i], s.source_id[i - 1]);
        }
        // Not covered by any earlier nucleus
        for (std::size_t j = 0; j < i; j++) {
            const double d = std::sqrt((s.x[i] - s.x[j]) * (s.x[i] - s.x[j]) +
                                       (s.y[i] - s.y[j]) * (s.y[i] - s.y[j]) +
                                       (s.z[i] - s.z[j]) * (s.z[i] - s.z[j]));
            EXPECT_GE(d, v * (s.nucleation_time[i] - s.nucleation_time[j]))
                << "nucleus " << i << " covered by " << j;
        }
    }

    // Discarded candidates: covered by the reported nucleus
    for (std::size_t r = 0; r < report.rejected_time.size(); r++) {
        const std::int64_t j = report.rejected_by[r];
        ASSERT_GE(j, 0);
        ASSERT_LT(j, static_cast<std::int64_t>(s.Size()));
        const double t = report.rejected_time[r];
        const double dx = report.rejected_xyz[3 * r] - s.x[j];
        const double dy = report.rejected_xyz[3 * r + 1] - s.y[j];
        const double dz = report.rejected_xyz[3 * r + 2] - s.z[j];
        EXPECT_LT(s.nucleation_time[j], t);
        EXPECT_LT(std::sqrt(dx * dx + dy * dy + dz * dz), v * (t - s.nucleation_time[j]))
            << "rejected candidate " << r;
    }

    // Completion bound: every vertex is reached before it
    EXPECT_TRUE(std::isfinite(report.completion_time_bound));
    EXPECT_GT(report.last_candidate_time, 0.0);
    for (int i = 0; i < serial.GetNV(); i++) {
        const double* p = serial.GetVertex(i);
        EXPECT_LE(ArrivalTime(s, v, p[0], p[1], p[2]), report.completion_time_bound);
    }

    // The arrival time T(x) = min_j (t_j + |x - s_j| / v) is not convex, so an element interior
    // can be reached later than all of its vertices. The bound must hold there too: check the
    // element centroid and 20 random points of every tet (uniform barycentric coordinates by
    // rejection from the unit cube, mapped with the element transformation).
    {
        std::mt19937_64 gen(99);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        double worst = -DBL_MAX;
        for (int e = 0; e < serial.GetNE(); e++) {
            mfem::ElementTransformation* trans = serial.GetElementTransformation(e);
            for (int k = 0; k <= 20; k++) {
                mfem::IntegrationPoint ip;
                if (k == 0) {
                    ip.Set3(0.25, 0.25, 0.25); // centroid of the reference tet
                } else {
                    double a, b, c;
                    do {
                        a = uni(gen);
                        b = uni(gen);
                        c = uni(gen);
                    } while (a + b + c > 1.0);
                    ip.Set3(a, b, c);
                }
                mfem::Vector x(3);
                trans->Transform(ip, x);
                const double arrival = ArrivalTime(s, v, x(0), x(1), x(2));
                worst = std::max(worst, arrival);
                EXPECT_LE(arrival, report.completion_time_bound) << "element " << e;
            }
        }
        EXPECT_GT(worst, 0.0);
    }

    // Direct check that stopping early changed nothing: regenerate the candidate stream past the
    // stop exactly as johnson_mehl.cpp does (candidate k: key (rng_seed, JOHNSON_MEHL, k);
    // sample 1 = exponential waiting time with mean 1 / (I V_box), sample 0 = position in the
    // sampling box), continuing from the last examined candidate. Every one of the next 20000
    // candidates that lies in the material must already be covered by an accepted nucleus, so
    // none of them could have been accepted had the loop continued.
    {
        double box_volume = 1.0;
        for (int d = 0; d < 3; d++) {
            box_volume *= (report.sampling_hi[d] - report.sampling_lo[d]);
        }
        const double inv_box_rate = 1.0 / (params.nucleation_rate * box_volume);
        double t = report.last_candidate_time;
        int in_material = 0;
        for (std::int64_t k = report.candidates_generated;
             k < report.candidates_generated + 20000; k++) {
            const ms::CounterRng rng(params.rng_seed, ms::RngPurpose::JOHNSON_MEHL,
                                     static_cast<std::uint64_t>(k));
            t += -std::log(1.0 - rng.UniformDouble(1)) * inv_box_rate;
            if (k == report.candidates_generated) {
                EXPECT_GT(t, report.completion_time_bound) << "the first unexamined candidate "
                                                              "should be born after the bound";
            }
            double u[3];
            ms::UniformUnitCube(rng, 0, u);
            double c[3];
            for (int d = 0; d < 3; d++) {
                const double extent = report.sampling_hi[d] - report.sampling_lo[d];
                c[d] = report.sampling_lo[d] + u[d] * extent;
            }
            if (!InLShape(c[0], c[1], c[2])) {
                continue;
            }
            in_material++;
            bool covered = false;
            for (std::size_t j = 0; j < s.Size() && !covered; j++) {
                const double d = std::sqrt((c[0] - s.x[j]) * (c[0] - s.x[j]) +
                                           (c[1] - s.y[j]) * (c[1] - s.y[j]) +
                                           (c[2] - s.z[j]) * (c[2] - s.z[j]));
                covered = d < v * (t - s.nucleation_time[j]);
            }
            EXPECT_TRUE(covered) << "candidate " << k << " after the stop is not covered";
        }
        EXPECT_GT(in_material, 10000);
    }

    // Each nucleus wins at its own position. For j born earlier, "i not covered by j" gives
    // |s_i - s_j| >= v (t_i - t_j), i.e. f_j(s_i) >= f_i(s_i); for j born later, "j not covered
    // by i" gives the same inequality with the roles swapped. Ties have probability zero.
    ms::SeedIndex<Exec, ms::AdditiveMetric> index;
    index.Build(s, ms::MetricSetup{}, 8, HostAllocator());
    const int n = static_cast<int>(s.Size());
    mfem::Vector qx(n), qy(n), qz(n);
    for (int i = 0; i < n; i++) {
        qx(i) = s.x[i];
        qy(i) = s.y[i];
        qz(i) = s.z[i];
    }
    mfem::Array<ms::BestPair> best;
    index.FindBest(qx, qy, qz, best, false);
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(best[i].id1, i) << "nucleus " << i << " does not own its own position";
    }
}

// Rank-count independence: the distributed run must reproduce, bitwise, the run each rank makes
// alone on the whole serial mesh (MPI_COMM_SELF) - positions, times, weights, ids and the report
// counts. Covered: hexes (Newton inside test) to full transformation, tets with an end time, and
// a time-dependent rate (thinning).
TEST(JohnsonMehl, RankIndependent) {
    struct Case {
        bool tets;
        double end_time;
        bool ramp;
    };
    const Case cases[] = {{false, 0.0, false}, {true, 0.08, false}, {true, 0.0, true}};
    for (const Case& c : cases) {
        mfem::Mesh serial = LShape(3, c.tets);
        auto pmesh = Distribute(serial);
        ms::JohnsonMehlParameters params = JmParams(1500.0, 9);
        params.end_time = c.end_time;
        if (c.ramp) {
            // Rate ramps up linearly from 0 to I_max over the first 0.1 time units
            params.rate_function = [](double t) { return 1500.0 * std::min(1.0, t / 0.1); };
        }
        ms::JohnsonMehlReport rep_par, rep_ref;
        const ms::SeedSet par = ms::GenerateJohnsonMehlSeeds(*pmesh, params, &rep_par);
        const ms::SeedSet ref =
            ms::GenerateJohnsonMehlSeeds(serial, MPI_COMM_SELF, params, &rep_ref);
        ASSERT_GT(ref.Size(), 5u);
        ExpectSameNuclei(par, ref);
        EXPECT_EQ(rep_par.candidates_generated, rep_ref.candidates_generated);
        EXPECT_EQ(rep_par.candidates_in_material, rep_ref.candidates_in_material);
        EXPECT_EQ(rep_par.rejected_covered, rep_ref.rejected_covered);
        EXPECT_EQ(rep_par.rejected_by_rate, rep_ref.rejected_by_rate);
        EXPECT_EQ(rep_par.stop, rep_ref.stop);
        if (c.end_time > 0.0) {
            EXPECT_EQ(rep_par.stop, ms::JohnsonMehlStop::END_TIME);
            for (double t : par.nucleation_time) {
                EXPECT_LE(t, c.end_time);
            }
        }
        if (c.ramp) {
            EXPECT_GT(rep_par.rejected_by_rate, 0);
        }
    }
}

// The stopping rules only cut the same candidate stream at different places: with num_nuclei = K
// the result is the first K nuclei of the full run, and with an end time T it is exactly the
// nuclei of the full run born at or before T. (The completion stop, used by the full run, is
// checked to be a proven bound in CoverageRulesAndCompletion.)
TEST(JohnsonMehl, StopRulesTakePrefixesOfOneStream) {
    mfem::Mesh serial = UnitCube(4, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    const ms::JohnsonMehlParameters full_params = JmParams(5000.0, 77);
    const ms::SeedSet full = ms::GenerateJohnsonMehlSeeds(*pmesh, full_params);
    ASSERT_GT(full.Size(), 40u);

    ms::JohnsonMehlParameters count_params = full_params;
    count_params.num_nuclei = 25;
    ms::JohnsonMehlReport count_report;
    const ms::SeedSet first = ms::GenerateJohnsonMehlSeeds(*pmesh, count_params, &count_report);
    EXPECT_EQ(count_report.stop, ms::JohnsonMehlStop::NUM_NUCLEI);
    ASSERT_EQ(first.Size(), 25u);
    for (std::size_t i = 0; i < first.Size(); i++) {
        EXPECT_EQ(first.source_id[i], full.source_id[i]);
        EXPECT_EQ(first.x[i], full.x[i]);
        EXPECT_EQ(first.nucleation_time[i], full.nucleation_time[i]);
    }

    ms::JohnsonMehlParameters time_params = full_params;
    time_params.end_time = full.nucleation_time[full.Size() / 2];
    const ms::SeedSet early = ms::GenerateJohnsonMehlSeeds(*pmesh, time_params);
    std::size_t expected = 0;
    while (expected < full.Size() && full.nucleation_time[expected] <= time_params.end_time) {
        expected++;
    }
    ASSERT_EQ(early.Size(), expected);
    for (std::size_t i = 0; i < early.Size(); i++) {
        EXPECT_EQ(early.source_id[i], full.source_id[i]);
    }
}

// Statistical check against the classical Johnson-Mehl grain density for constant rate,
// N_V = Gamma(5/4) (3/pi)^(1/4) (I/v)^(3/4) ~= 0.896 (I/v)^(3/4) in infinite space. In a finite
// cube the grains touching the surface can only grow inward, so the surface layer transforms
// later and collects extra nuclei; the count is therefore expected somewhat above the
// infinite-space value. Averaged over 4 RNG keys with about 300 predicted grains each, it must
// lie between 0.9 and 1.4 times the prediction.
TEST(JohnsonMehl, GrainDensityNearClassicalValue) {
    mfem::Mesh serial = UnitCube(6, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    const double target = 300.0; // grains in the unit cube (infinite-space prediction)
    const double rate = std::pow(target / 0.8960, 4.0 / 3.0); // v = 1
    const double predicted = ms::JohnsonMehlGrainDensity(rate, 1.0);
    EXPECT_NEAR(predicted, target, 0.1);
    double total = 0.0;
    for (std::uint64_t key = 0; key < 4; key++) {
        const ms::SeedSet s = ms::GenerateJohnsonMehlSeeds(*pmesh, JmParams(rate, key));
        total += static_cast<double>(s.Size());
    }
    const double ratio = total / (4.0 * predicted);
    if (Rank() == 0) {
        std::cout << "[ Johnson-Mehl ] mean grains " << total / 4.0 << ", infinite-space "
                  << "prediction " << predicted << ", ratio " << ratio << std::endl;
    }
    EXPECT_GT(ratio, 0.9);
    EXPECT_LT(ratio, 1.4);
}

// Invalid parameters are rejected identically on every rank (no hang), and a rate function that
// exceeds the declared maximum rate is detected.
TEST(JohnsonMehl, InvalidParametersRejected) {
    mfem::Mesh serial = UnitCube(2, mfem::Element::TETRAHEDRON);
    auto pmesh = Distribute(serial);
    ms::JohnsonMehlParameters p = JmParams(100.0, 1);
    p.nucleation_rate = 0.0;
    EXPECT_THROW(ms::GenerateJohnsonMehlSeeds(*pmesh, p), std::invalid_argument);
    p = JmParams(100.0, 1);
    p.growth_speed = -1.0;
    EXPECT_THROW(ms::GenerateJohnsonMehlSeeds(*pmesh, p), std::invalid_argument);
    p = JmParams(100.0, 1);
    p.end_time = -1.0;
    EXPECT_THROW(ms::GenerateJohnsonMehlSeeds(*pmesh, p), std::invalid_argument);
    p = JmParams(100.0, 1);
    p.use_sampling_box = true; // empty box
    EXPECT_THROW(ms::GenerateJohnsonMehlSeeds(*pmesh, p), std::invalid_argument);
    p = JmParams(100.0, 1);
    p.rate_function = [](double) { return 200.0; }; // above nucleation_rate
    EXPECT_THROW(ms::GenerateJohnsonMehlSeeds(*pmesh, p), std::invalid_argument);
    // A sampling box outside the material never yields a nucleus: stall error, or partial result
    p = JmParams(100.0, 1);
    p.use_sampling_box = true;
    p.sampling_box_lo = {5.0, 5.0, 5.0};
    p.sampling_box_hi = {6.0, 6.0, 6.0};
    p.max_candidates = 5000;
    EXPECT_THROW(ms::GenerateJohnsonMehlSeeds(*pmesh, p), std::runtime_error);
    p.allow_stall = true;
    ms::JohnsonMehlReport report;
    EXPECT_EQ(ms::GenerateJohnsonMehlSeeds(*pmesh, p, &report).Size(), 0u);
    EXPECT_EQ(report.stop, ms::JohnsonMehlStop::STALLED);
    EXPECT_EQ(report.candidates_generated, 5000);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
