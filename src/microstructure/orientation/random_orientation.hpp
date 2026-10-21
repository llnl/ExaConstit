#pragma once

#include "microstructure/util/counter_rng.hpp"
#include "microstructure/util/random_distributions.hpp"

#include "mfem.hpp"

#include <cstdint>
#include <math.h>

/**
 * @file random_orientation.hpp
 * @brief Uniformly distributed random crystal orientations (unit quaternions) from the
 * counter-based RNG.
 *
 * Header-only and `MFEM_HOST_DEVICE`, so usable inside mfem::forall kernels over grains. The
 * result for (rng_seed, RngPurpose::ORIENTATION, grain id) is independent of rank count. Output
 * follows ExaConstit's quaternion convention, `w x y z` order; a uniform random rotation has no
 * preferred frame, so crystal-to-sample and sample-to-crystal readings are equally uniform.
 */

namespace exaconstit::microstructure {

/**
 * @brief Uniformly distributed rotation as a unit quaternion (Shoemake, "Uniform random
 * rotations", Graphics Gems III, 1992), written `w x y z` with w >= 0.
 *
 * Shoemake's method maps three uniforms u0, u1, u2 to a point on the unit 3-sphere such that
 * rotations are uniform (every orientation equally likely; the Haar measure): with
 * r1 = sqrt(1 - u0), r2 = sqrt(u0),
 * q = (r2 cos 2 pi u2, r1 sin 2 pi u1, r1 cos 2 pi u1, r2 sin 2 pi u2). q and -q are the same
 * rotation, so the sign is flipped when w < 0 to make the output canonical.
 *
 * Reads the same bits as UniformUnitCube(rng, sample, ...). Host and device; never fails.
 *
 * @param rng    stream of the grain (normally purpose ORIENTATION, id = global grain id).
 * @param sample sample index (0 unless several orientations per grain are needed).
 * @param q      the quaternion, written to q[0..3] as w x y z (unit norm up to round-off).
 */
MFEM_HOST_DEVICE inline void RandomQuaternion(const CounterRng& rng, std::uint32_t sample,
                                              double q[4]) {
    // Three uniforms from blocks (sample, 0..1)
    double u[3];
    UniformUnitCube(rng, sample, u);

    // Shoemake's map from the unit cube to the unit 3-sphere
    const double r1 = sqrt(1.0 - u[0]);
    const double r2 = sqrt(u[0]);
    const double angle1 = detail::TWO_PI * u[1];
    const double angle2 = detail::TWO_PI * u[2];
    q[0] = r2 * cos(angle2);
    q[1] = r1 * sin(angle1);
    q[2] = r1 * cos(angle1);
    q[3] = r2 * sin(angle2);

    // Canonical sign: w >= 0
    if (q[0] < 0.0) {
        for (int k = 0; k < 4; ++k) {
            q[k] = -q[k];
        }
    }
}

} // namespace exaconstit::microstructure
