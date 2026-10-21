/*
 * Tests of orientation/random_orientation.hpp: uniformly distributed random crystal orientations
 * (unit quaternions, w x y z, canonical sign w >= 0) from the counter-based RNG. Serial (runs
 * identically on every rank).
 */
#include "microstructure/orientation/random_orientation.hpp"
#include "microstructure/util/counter_rng.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

namespace ms = exaconstit::microstructure;

// Random orientations are unit quaternions with the canonical sign w >= 0 and are uniformly
// distributed over rotations. For a uniform rotation the scalar part w = cos(theta/2) has density
// proportional to sqrt(1 - w^2) on [-1, 1], so E|w| = 4/(3 pi) ~ 0.4244 and
// E|(x, y, z)| = E[sin(theta/2)] = 8/(3 pi) ~ 0.8488; the vector components are symmetric about 0.
TEST(RandomOrientation, QuaternionsAreUnitCanonicalAndUniform) {
    const ms::CounterRng rng(2026u, ms::RngPurpose::ORIENTATION, 0u);
    const int n = 200000;
    double sum_w = 0.0;
    double sum_v = 0.0;
    double sum_xyz[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; ++i) {
        double q[4];
        ms::RandomQuaternion(rng, static_cast<std::uint32_t>(i), q);
        const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        ASSERT_NEAR(norm, 1.0, 1e-14);
        ASSERT_GE(q[0], 0.0);
        sum_w += q[0];
        sum_v += std::sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (int k = 0; k < 3; ++k) {
            sum_xyz[k] += q[k + 1];
        }
    }
    const double pi = 3.14159265358979323846; // M_PI is not standard C++
    EXPECT_NEAR(sum_w / n, 4.0 / (3.0 * pi), 3e-3);
    EXPECT_NEAR(sum_v / n, 8.0 / (3.0 * pi), 3e-3);
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(sum_xyz[k] / n, 0.0, 5e-3);
    }
}

// RandomQuaternion compiles and runs inside an mfem::forall kernel and gives the same quaternions
// as host calls. Locally this runs on the CPU (bitwise comparison is valid there; on a GPU,
// sin/cos may differ from the host in the last bits).
TEST(RandomOrientation, KernelMatchesHost) {
    const int n = 257;
    mfem::Vector out(4 * n);
    const std::uint64_t seed = 77u;

    // Kernel: one random orientation per grain id, out[4*i + k] with k -> w x y z
    double* d_out = out.Write();
    mfem::forall(n, [=] MFEM_HOST_DEVICE(int i) {
        const ms::CounterRng rng(seed, ms::RngPurpose::ORIENTATION, static_cast<std::uint64_t>(i));
        ms::RandomQuaternion(rng, 0u, d_out + 4 * i);
    });

    // Host: the same ids, visited backwards to also exercise order independence
    const double* h_out = out.HostRead();
    for (int i = n - 1; i >= 0; --i) {
        const ms::CounterRng rng(seed, ms::RngPurpose::ORIENTATION, static_cast<std::uint64_t>(i));
        double q[4];
        ms::RandomQuaternion(rng, 0u, q);
        for (int k = 0; k < 4; ++k) {
            EXPECT_EQ(h_out[4 * i + k], q[k]);
        }
    }
}
