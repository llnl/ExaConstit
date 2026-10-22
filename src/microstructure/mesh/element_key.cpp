#include "microstructure/mesh/element_key.hpp"

#include <algorithm>
#include <array>
#include <cstring>

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

} // namespace

/*
 * ElementKey
 *
 * Steps:
 *  1. Copy the element's vertex coordinates (Mesh::GetElementVertices, then Mesh::GetVertex, which
 *     returns a pointer to the 3 stored coordinates of a vertex).
 *  2. Sort them lexicographically (std::array's operator< compares x, then y, then z), so the
 *     result does not depend on the element's local vertex order, which can differ between the
 *     serial mesh, a partition and a face-neighbor copy.
 *  3. Fold the raw bit pattern of every coordinate into the running hash: h = Mix64(h ^ bits).
 *     std::memcpy is the portable way to read a double's bits as an integer.
 *
 * This must stay bit-identical to ElementKey in seeding/seed_generators.cpp (same start value,
 * same order, same mixer), so that both give the same key for the same element.
 */
std::uint64_t ElementKey(const mfem::Mesh& mesh, int e) {
    // Step 1: vertex coordinates of element e
    mfem::Array<int> vertices;
    mesh.GetElementVertices(e, vertices);
    std::vector<std::array<double, 3>> coords(static_cast<std::size_t>(vertices.Size()));
    for (int v = 0; v < vertices.Size(); v++) {
        const double* x = mesh.GetVertex(vertices[v]);
        coords[static_cast<std::size_t>(v)] = {x[0], x[1], x[2]};
    }

    // Step 2: order-independent vertex list
    std::sort(coords.begin(), coords.end());

    // Step 3: fold the exact coordinate bits into the hash
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
