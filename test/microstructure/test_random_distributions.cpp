/*
 * Tests of util/random_distributions.hpp: uniform on [a, b], uniform in the unit cube, normal and
 * log-normal samples built on the counter-based RNG. Each test checks the defining moments of the
 * distribution with tolerances of several standard errors, and that the functions run inside
 * mfem::forall kernels. Serial (runs identically on every rank).
 */
#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/random_distributions.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

namespace ms = exaconstit::microstructure;

// UniformIn(a, b) on [-2, 3] has mean (a + b)/2 = 0.5 and stays in range; each component of
// UniformUnitCube has mean 0.5. Tolerances ~6 standard errors for n = 200000 (standard error of
// UniformIn's mean: 5 * 0.29 / sqrt(n) ~ 3.2e-3; of a unit uniform: 6.5e-4).
TEST(RandomDistributions, UniformInAndUnitCube) {
    const ms::CounterRng rng(12345u, ms::RngPurpose::GENERIC, 7u);
    const int n = 200000;
    double sum_ab = 0.0;
    double sum3[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        const auto s = static_cast<std::uint32_t>(i);
        const double x = ms::UniformIn(rng, -2.0, 3.0, s);
        EXPECT_GE(x, -2.0);
        EXPECT_LE(x, 3.0); // [a, b] up to round-off
        sum_ab += x;
        double u[3];
        ms::UniformUnitCube(rng, s, u);
        for (int k = 0; k < 3; ++k) {
            sum3[k] += u[k];
        }
    }
    EXPECT_NEAR(sum_ab / n, 0.5, 2e-2);
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(sum3[k] / n, 0.5, 4e-3);
    }
}

// Box-Muller normals have mean 0 and variance 1. LogNormal(mu, sigma) is exp(mu + sigma * z)
// built from the same z as StandardNormal(rng, sample), so log(LogNormal) averages to exactly
// mu + sigma * mean(z) up to round-off; this checks that mu and sigma are the parameters of the
// underlying normal, as documented.
TEST(RandomDistributions, NormalAndLogNormalMoments) {
    const ms::CounterRng rng(99u, ms::RngPurpose::GENERIC, 0u);
    const int n = 200000;
    double sum = 0.0;
    double sum_sq = 0.0;
    double sum_log = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto s = static_cast<std::uint32_t>(i);
        const double z = ms::StandardNormal(rng, s);
        ASSERT_TRUE(std::isfinite(z));
        sum += z;
        sum_sq += z * z;
        const double ln = ms::LogNormal(rng, 1.0, 0.5, s);
        ASSERT_GT(ln, 0.0);
        sum_log += std::log(ln);
    }
    const double mean = sum / n;
    EXPECT_NEAR(mean, 0.0, 1.5e-2);
    EXPECT_NEAR(sum_sq / n - mean * mean, 1.0, 2e-2);
    EXPECT_NEAR(sum_log / n, 1.0 + 0.5 * mean, 1e-10);
}

// The distributions compile and run inside an mfem::forall kernel (MFEM_HOST_DEVICE, global
// ::log/::cos/::exp) and give the same values as host calls. Locally this runs on the CPU; on a
// GPU the transcendental functions may differ from the host in the last bits, so this test
// compares exactly only because both sides run on the host here.
TEST(RandomDistributions, KernelMatchesHost) {
    const int n = 257;
    mfem::Vector out(5 * n);
    const std::uint64_t seed = 11u;

    // Layout out[5*i + k]: k = 0..2 unit cube, 3 normal, 4 log-normal
    double* d_out = out.Write();
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int i) {
        const ms::CounterRng rng(seed, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        ms::UniformUnitCube(rng, 0u, d_out + 5 * i);
        d_out[5 * i + 3] = ms::Normal(rng, 1.0, 2.0, 1u);
        d_out[5 * i + 4] = ms::LogNormal(rng, 0.0, 0.5, 2u);
    });

    const double* h_out = out.HostRead();
    for (int i = 0; i < n; ++i) {
        const ms::CounterRng rng(seed, ms::RngPurpose::GENERIC, static_cast<std::uint64_t>(i));
        double u[3];
        ms::UniformUnitCube(rng, 0u, u);
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(h_out[5 * i + k], u[k]);
        }
        EXPECT_EQ(h_out[5 * i + 3], ms::Normal(rng, 1.0, 2.0, 1u));
        EXPECT_EQ(h_out[5 * i + 4], ms::LogNormal(rng, 0.0, 0.5, 2u));
    }
}
