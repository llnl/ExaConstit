/*
 * Implementation of element_key.hpp: corner positions from the true geometry, a power-of-two
 * quantum from the element's size and coordinate magnitude, rounding to that grid, and a
 * splitmix64 hash of the sorted rounded corners.
 *
 * Terms used in this file:
 * - Quantum q: the grid spacing coordinates are rounded to before hashing.
 * - ulp (unit in the last place): the spacing of doubles near a value, i.e. its last-bit size.
 */
#include "microstructure/mesh/element_key.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "microstructure/mesh/straight_sided.hpp"

namespace exaconstit::microstructure {

namespace {

/*
 * Mix64: the splitmix64 finalizer (S. Vigna, "splitmix64", public domain). It is a fixed,
 * one-to-one scrambling of a 64-bit integer: every input bit influences every output bit, so
 * nearby inputs (coordinates that differ in one bit) give unrelated outputs. Two xor-shift /
 * multiply rounds with the published constants, then a final xor-shift.
 */
std::uint64_t Mix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/*
 * CornerPositions: the corner positions of element e (Mesh::GetElementVertices, then VertexPoint
 * for each vertex: the Nodes value at the vertex dof for continuous Nodes, else Mesh::GetVertex;
 * see mesh/straight_sided.hpp), three components each (missing ones are 0). Only this rank's own
 * copy is read: keys and seeding positions of an element are computed by its owner only.
 */
std::vector<std::array<double, 3>> CornerPositions(const mfem::Mesh& mesh, int e) {
    mfem::Array<int> vertices;
    mesh.GetElementVertices(e, vertices);
    const int sdim = mesh.SpaceDimension();
    std::vector<std::array<double, 3>> coords(static_cast<std::size_t>(vertices.Size()));
    double x[3];
    for (int v = 0; v < vertices.Size(); v++) {
        x[0] = x[1] = x[2] = 0.0;
        VertexPoint(mesh, vertices[v], x);
        std::array<double, 3>& c = coords[static_cast<std::size_t>(v)];
        for (int d = 0; d < 3; d++) {
            c[static_cast<std::size_t>(d)] = (d < sdim) ? x[d] : 0.0;
        }
    }
    return coords;
}

/*
 * QuantumFromCorners: q = 2^(floor(log2(c s)) - ELEMENT_KEY_QUANTUM_BITS) with the length scale
 * s = max(h, min(M, ELEMENT_KEY_MAX_MAGNITUDE_RATIO h)): h = largest edge of the box around the
 * corners, M = largest |corner coordinate|, c = ELEMENT_KEY_SIZE_FACTOR (file comment of
 * element_key.hpp). std::frexp writes c s = m 2^E with m in [0.5, 1), so floor(log2(c s)) =
 * E - 1, and std::ldexp builds the power of two exactly. A scale that is zero, negative or not
 * finite (degenerate element at the origin, NaN coordinates) returns 0: no rounding.
 */
double QuantumFromCorners(const std::vector<std::array<double, 3>>& coords) {
    double size = 0.0;
    double magnitude = 0.0;
    for (std::size_t d = 0; d < 3; d++) {
        double lo = coords.empty() ? 0.0 : coords[0][d];
        double hi = lo;
        for (const auto& c : coords) {
            lo = std::min(lo, c[d]);
            hi = std::max(hi, c[d]);
            magnitude = std::max(magnitude, std::fabs(c[d]));
        }
        size = std::max(size, hi - lo);
    }
    // The coordinate magnitude counts (round-off between copies grows with it), but at most
    // ELEMENT_KEY_MAX_MAGNITUDE_RATIO element sizes, so q stays far below the element size
    const double scale = std::max(size,
                                  std::min(magnitude, ELEMENT_KEY_MAX_MAGNITUDE_RATIO * size));
    const double scaled = ELEMENT_KEY_SIZE_FACTOR * scale;
    if (!(scaled > 0.0) || !std::isfinite(scaled)) {
        return 0.0;
    }
    int exponent = 0;
    std::frexp(scaled, &exponent);
    return std::ldexp(1.0, exponent - 1 - ELEMENT_KEY_QUANTUM_BITS);
}

} // namespace

/*
 * ElementKeyQuantum: QuantumFromCorners of the element's own corner positions.
 */
double ElementKeyQuantum(const mfem::Mesh& mesh, int e) {
    return QuantumFromCorners(CornerPositions(mesh, e));
}

/*
 * RoundToKeyQuantum: x rounded to the nearest multiple of q (ties to even, std::nearbyint in the
 * default rounding mode). Division and multiplication by a power of two are exact (no rounding)
 * unless the result leaves the normal range, so the result is exactly k q for the integer
 * k = nearbyint(x / q). For |x / q| >= 2^53 the quotient is already an integer and x comes back
 * unchanged. Adding +0.0 turns a -0.0 result (a small negative x) into +0.0, so both signs of
 * zero give the same bits. q = 0 (degenerate element) leaves x unchanged.
 */
double RoundToKeyQuantum(double x, double q) {
    if (!(q > 0.0)) {
        return x + 0.0;
    }
    const double rounded = std::nearbyint(x / q) * q;
    if (!std::isfinite(rounded)) {
        return x + 0.0; // overflow of x / q (q tiny relative to x): keep the exact value
    }
    return rounded + 0.0;
}

/*
 * RoundIntervalOutward: [lo, hi] -> [(nearbyint(lo / q) - 1) q, (nearbyint(hi / q) + 1) q].
 * Rounding to the nearest multiple first puts the decision boundaries at half-multiples of q
 * (as for the key), not at the multiples themselves: a corner that sits exactly on a grid point
 * (any dyadic coordinate, e.g. 0.5) and has a copy one ulp below it would otherwise be floored
 * to different grid points. The nearest multiple is within q/2 of the input, so subtracting
 * (adding) one more q keeps the interval containing [lo, hi]. q = 0 leaves both unchanged.
 */
void RoundIntervalOutward(double q, double& lo, double& hi) {
    if (!(q > 0.0)) {
        return;
    }
    const double new_lo = (std::nearbyint(lo / q) - 1.0) * q;
    const double new_hi = (std::nearbyint(hi / q) + 1.0) * q;
    if (std::isfinite(new_lo) && std::isfinite(new_hi)) {
        lo = new_lo + 0.0;
        hi = new_hi + 0.0;
    }
}

/*
 * ElementKey
 *
 * Steps:
 *  1. The element's corner positions (CornerPositions; this rank's own copy).
 *  2. The quantum q from the element's size and coordinate magnitude (QuantumFromCorners).
 *  3. Round every coordinate to the nearest multiple of q (RoundToKeyQuantum).
 *  4. Sort the rounded corners lexicographically (std::array's operator< compares x, then y,
 *     then z), so the result does not depend on the element's local vertex order, which can
 *     differ between the serial mesh, a partition and a face-neighbor copy.
 *  5. Fold the bit pattern of every rounded coordinate into the running hash: h = Mix64(h ^ bits).
 *     std::memcpy is the portable way to read a double's bits as an integer.
 *
 * seeding/seed_generators.cpp uses this function directly for element-weighted sampling.
 */
std::uint64_t ElementKey(const mfem::Mesh& mesh, int e) {
    // Steps 1-2: corners and quantum
    std::vector<std::array<double, 3>> coords = CornerPositions(mesh, e);
    const double q = QuantumFromCorners(coords);

    // Step 3: round every coordinate to the grid
    for (auto& c : coords) {
        for (double& value : c) {
            value = RoundToKeyQuantum(value, q);
        }
    }

    // Step 4: order-independent corner list
    std::sort(coords.begin(), coords.end());

    // Step 5: fold the rounded coordinate bits into the hash
    std::uint64_t h = 0x9E3779B97F4A7C15ULL; // arbitrary nonzero start value
    for (const auto& c : coords) {
        for (int d = 0; d < 3; d++) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &c[static_cast<std::size_t>(d)], sizeof(bits));
            h = Mix64(h ^ bits);
        }
    }
    return h;
}

/*
 * ComputeElementKeys: ElementKey for every local element, in local element order.
 */
std::vector<std::uint64_t> ComputeElementKeys(const mfem::Mesh& mesh) {
    const int ne = mesh.GetNE();
    std::vector<std::uint64_t> keys(static_cast<std::size_t>(ne));
    for (int e = 0; e < ne; e++) {
        keys[static_cast<std::size_t>(e)] = ElementKey(mesh, e);
    }
    return keys;
}

} // namespace exaconstit::microstructure
