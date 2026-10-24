/**
 * @file test_node_labels.cpp
 * @brief Tests of conformal/node_labels (Stage 2a): label sets, boundary surfaces and constraint
 * counts of the mesh nodes.
 *
 * What is checked, and why:
 *  - Every node of the distributed mesh against a serial reference computed in the test from the
 *    serial mesh (label set = labels of the elements containing the node; boundary surfaces of a
 *    unit box = how many coordinates are 0 or 1), matched by coordinates. The reference does not
 *    depend on the partition, so passing at np = 1, 2, 3, 4 shows rank invariance. The meshes are
 *    distributed round robin (element e on rank e mod np), so most nodes are shared by several
 *    ranks and the exchange is exercised far more than with a compact partition.
 *  - The global summary against counts of the reference, and the exact class counts of simple
 *    configurations: a bicrystal (face nodes on and off the box faces and edges), a tricrystal on
 *    top of a fourth grain (triple lines and one quad point), four quadrants (a line of 4-grain
 *    nodes that is over-constrained where it meets the box).
 *  - Hex and tet meshes, and order-2 Nodes (every nodal point, including edge and face dofs).
 *  - The feature angle: a roof-shaped top face whose crease is 20.4 degrees is one surface with
 *    a 30-degree feature angle and two with 15 degrees.
 *  - Invalid input throws on every rank.
 *
 * Assertions use EXPECT_* (not ASSERT_*) wherever a later collective call follows: an ASSERT
 * returns from the test on one rank only, and the other ranks would then wait forever.
 */
#include "microstructure/conformal/node_labels.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mpi.h>
#include <set>
#include <stdexcept>
#include <vector>

namespace ms = exaconstit::microstructure;

namespace {

using Key = std::array<long long, 3>;
using LabelFunction = std::function<int(const double*)>;

int WorldSize() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

/** Coordinates rounded to 1e-9, so copies that differ in the last bits get the same key. */
Key MakeKey(const double* x) {
    return {std::llround(x[0] * 1e9), std::llround(x[1] * 1e9), std::llround(x[2] * 1e9)};
}

/** Number of coordinates of x on the faces of the unit box. */
int BoxSurfaces(const double* x) {
    int count = 0;
    for (int d = 0; d < 3; ++d) {
        count += (std::fabs(x[d]) < 1e-12 || std::fabs(x[d] - 1.0) < 1e-12) ? 1 : 0;
    }
    return count;
}

/** Element labels of `mesh` from the label function at the element centers. */
mfem::Array<int> CenterLabels(mfem::Mesh& mesh, const LabelFunction& label) {
    mfem::Array<int> labels(mesh.GetNE());
    mfem::Vector center(3);
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mesh.GetElementCenter(e, center);
        labels[e] = label(center.GetData());
    }
    return labels;
}

/** Reference per node: label set and position. */
struct RefNode {
    std::set<int> labels;
    std::array<double, 3> x{};
};

/**
 * Serial reference: for every node of the serial mesh (its Nodes dofs if it has Nodes, else its
 * vertices) the labels of the elements containing it, keyed by coordinates.
 */
std::map<Key, RefNode> SerialReference(mfem::Mesh& mesh, const mfem::Array<int>& labels) {
    std::map<Key, RefNode> ref;
    mfem::Array<int> nodes;
    const mfem::GridFunction* gf = mesh.GetNodes();
    for (int e = 0; e < mesh.GetNE(); ++e) {
        if (gf != nullptr) {
            gf->FESpace()->GetElementDofs(e, nodes);
        } else {
            mesh.GetElementVertices(e, nodes);
        }
        for (int j = 0; j < nodes.Size(); ++j) {
            double x[3];
            for (int d = 0; d < 3; ++d) {
                x[d] = (gf != nullptr) ? (*gf)(gf->FESpace()->DofToVDof(nodes[j], d))
                                       : mesh.GetVertex(nodes[j])[d];
            }
            RefNode& node = ref[MakeKey(x)];
            node.labels.insert(labels[e]);
            node.x = {x[0], x[1], x[2]};
        }
    }
    return ref;
}

/** Element e on rank e mod np. */
std::unique_ptr<mfem::ParMesh> Distribute(mfem::Mesh& serial) {
    std::vector<int> part(static_cast<std::size_t>(serial.GetNE()));
    for (int e = 0; e < serial.GetNE(); ++e) {
        part[static_cast<std::size_t>(e)] = e % WorldSize();
    }
    return std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial, part.data());
}

/** Expected boundary surfaces of a reference node. */
using SurfaceFunction = std::function<int(const double*)>;

/** A distributed mesh and its node labels (declared in this order: the labels go first). */
struct LabeledMesh {
    std::unique_ptr<mfem::ParMesh> pmesh;
    std::unique_ptr<ms::NodeLabels> nl;
};

/**
 * Builds NodeLabels on the distributed mesh and compares every local node with the serial
 * reference (label set, boundary surfaces), then the summary with counts of the reference.
 */
LabeledMesh CheckAgainstReference(mfem::Mesh& serial, const LabelFunction& label,
                                  const SurfaceFunction& surfaces,
                                  const ms::NodeLabelOptions& options = {}) {
    const mfem::Array<int> serial_labels = CenterLabels(serial, label);
    const std::map<Key, RefNode> ref = SerialReference(serial, serial_labels);
    LabeledMesh run;
    run.pmesh = Distribute(serial);
    mfem::Array<int> labels = CenterLabels(*run.pmesh, label);
    run.nl = std::make_unique<ms::NodeLabels>(*run.pmesh, labels, options);
    const ms::NodeLabels* nl = run.nl.get();

    const double* xyz = nl->Field().Coordinates().HostRead();
    const int* off = nl->LabelOffsets().HostRead();
    const int* items = nl->Labels().HostRead();
    int mismatches = 0;
    for (int v = 0; v < nl->NumNodes(); ++v) {
        const auto it = ref.find(MakeKey(xyz + 3 * v));
        if (it == ref.end()) {
            ++mismatches;
            continue;
        }
        const std::set<int> got(items + off[v], items + off[v + 1]);
        const bool sorted = std::is_sorted(items + off[v], items + off[v + 1]) &&
                            static_cast<int>(got.size()) == nl->NumLabels(v);
        if (got != it->second.labels || !sorted ||
            nl->BoundarySurfaces(v) != surfaces(it->second.x.data())) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0);

    // Summary against the reference (every reference node once)
    ms::NodeLabelSummary expect;
    for (const auto& [key, node] : ref) {
        const int n = static_cast<int>(node.labels.size());
        const int b = std::min(surfaces(node.x.data()), 3);
        ++expect.count[static_cast<std::size_t>(std::min(n, 5) - 1)][static_cast<std::size_t>(b)];
        expect.over_constrained += (n - 1 + b > 3) ? 1 : 0;
        expect.max_labels = std::max(expect.max_labels, n);
        ++expect.num_nodes;
    }
    const ms::NodeLabelSummary& got = nl->Summary();
    EXPECT_EQ(got.count, expect.count);
    EXPECT_EQ(got.over_constrained, expect.over_constrained);
    EXPECT_EQ(got.max_labels, expect.max_labels);
    EXPECT_EQ(got.num_nodes, expect.num_nodes);
    return run;
}

/** Global number of local nodes at x satisfying pred (owned copies only). */
std::int64_t CountOwned(const ms::NodeLabels& nl,
                        const std::function<bool(const double*, int)>& pred) {
    const double* xyz = nl.Field().Coordinates().HostRead();
    const int* owned = nl.Field().OwnedVertexMarker().HostRead();
    std::int64_t count = 0;
    for (int v = 0; v < nl.NumNodes(); ++v) {
        count += (owned[v] != 0 && pred(xyz + 3 * v, v)) ? 1 : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    return count;
}

constexpr int FACE = static_cast<int>(ms::JunctionKind::FACE);
constexpr int QUAD = static_cast<int>(ms::JunctionKind::QUAD_POINT);

} // namespace

TEST(NodeLabels, HexBicrystal) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
    LabeledMesh run = CheckAgainstReference(
        serial, [](const double* x) { return x[0] < 0.5 ? 1 : 2; }, BoxSurfaces);
    const auto& nl = run.nl;
    // The 5 x 5 nodes of the plane x = 0.5: 9 inside, 12 on a box face, 4 on a box edge
    const auto& s = nl->Summary();
    EXPECT_EQ(s.count[FACE][0], 9);
    EXPECT_EQ(s.count[FACE][1], 12);
    EXPECT_EQ(s.count[FACE][2], 4);
    EXPECT_EQ(s.over_constrained, 0);
    EXPECT_EQ(s.max_labels, 2);
}

TEST(NodeLabels, TricrystalOnFourthGrain) {
    // z < 1/2: grain 1; above, three 120-degree sectors around the vertical line x = y = 1/2
    const double pi = 3.14159265358979323846;
    const auto label = [pi](const double* x) {
        if (x[2] < 0.5) {
            return 1;
        }
        const double angle = std::atan2(x[1] - 0.5, x[0] - 0.5);
        return 2 + static_cast<int>(std::floor((angle + pi) / (2.0 * pi / 3.0)));
    };
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(6, 6, 6, mfem::Element::HEXAHEDRON);
    LabeledMesh run = CheckAgainstReference(serial, label, BoxSurfaces);
    const auto& nl = run.nl;
    // One quad point, at the center
    const auto& s = nl->Summary();
    EXPECT_EQ(s.count[QUAD][0], 1);
    EXPECT_EQ(s.count[QUAD][1] + s.count[QUAD][2] + s.count[QUAD][3], 0);
    EXPECT_EQ(CountOwned(*nl,
                         [&](const double* x, int v) {
                             return std::fabs(x[0] - 0.5) + std::fabs(x[1] - 0.5) +
                                            std::fabs(x[2] - 0.5) < 1e-12 &&
                                    nl->Kind(v) == ms::JunctionKind::QUAD_POINT;
                         }),
              1);
}

TEST(NodeLabels, QuadrantsOverConstrainedAtTheBox) {
    // Four quadrants: the vertical line x = y = 1/2 has 4 labels; its two ends on the bottom
    // and top faces have 3 + 1 constraints
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
    LabeledMesh run = CheckAgainstReference(
        serial,
        [](const double* x) { return 1 + (x[0] > 0.5 ? 1 : 0) + (x[1] > 0.5 ? 2 : 0); },
        BoxSurfaces);
    const auto& s = run.nl->Summary();
    EXPECT_EQ(s.count[QUAD][0], 3);
    EXPECT_EQ(s.count[QUAD][1], 2);
    EXPECT_EQ(s.over_constrained, 2);
    EXPECT_EQ(s.max_labels, 4);
}

TEST(NodeLabels, TetInclinedBicrystal) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON);
    CheckAgainstReference(
        serial, [](const double* x) { return x[0] + 0.5 * x[1] + 0.25 * x[2] < 0.9 ? 3 : 7; },
        BoxSurfaces);
}

TEST(NodeLabels, OrderTwoNodes) {
    for (const auto type : {mfem::Element::HEXAHEDRON, mfem::Element::TETRAHEDRON}) {
        mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(3, 3, 3, type);
        serial.SetCurvature(2);
        LabeledMesh run = CheckAgainstReference(
            serial, [](const double* x) { return x[0] + 0.3 * x[2] < 0.6 ? 1 : 2; },
            BoxSurfaces);
        EXPECT_TRUE(run.nl->Field().UsesMeshNodes());
    }
}

TEST(NodeLabels, FeatureAngle) {
    // Top face z = 1 + 0.18 |x - 1/2|: two planes tilted by atan(0.18) = 10.2 degrees each, a
    // 20.4-degree crease along x = 1/2. All other faces stay planar.
    const auto roof = [](mfem::Mesh& mesh) {
        mesh.Transform([](const mfem::Vector& x, mfem::Vector& y) {
            y = x;
            y(2) = x(2) * (1.0 + 0.18 * std::fabs(x(0) - 0.5));
        });
    };
    const auto on_crease = [](const double* x) {
        return std::fabs(x[0] - 0.5) < 1e-12 && std::fabs(x[2] - 1.0) < 1e-12;
    };
    const auto top = [](const double* x) {
        return x[2] > 0.5 && std::fabs(x[2] - (1.0 + 0.18 * std::fabs(x[0] - 0.5))) < 1e-12;
    };
    // Box rule with the roof as the top face
    const auto roof_surfaces = [&](const double* x, bool split) {
        int count = 0;
        for (int d = 0; d < 2; ++d) {
            count += (std::fabs(x[d]) < 1e-12 || std::fabs(x[d] - 1.0) < 1e-12) ? 1 : 0;
        }
        count += (std::fabs(x[2]) < 1e-12 || top(x)) ? 1 : 0;
        return count + ((split && on_crease(x)) ? 1 : 0);
    };
    for (const double angle : {30.0, 15.0}) {
        mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
        roof(serial);
        ms::NodeLabelOptions options;
        options.feature_angle_degrees = angle;
        const bool split = angle < 20.0;
        LabeledMesh run = CheckAgainstReference(
            serial, [](const double*) { return 1; },
            [&](const double* x) { return roof_surfaces(x, split); }, options);
        const auto& nl = run.nl;
        // The 5 crease nodes: 3 inside the top face, 2 on the side faces y = 0, 1
        EXPECT_EQ(CountOwned(*nl,
                             [&](const double* x, int v) {
                                 return on_crease(x) &&
                                        nl->BoundarySurfaces(v) == (split ? 2 : 1);
                             }),
                  3);
    }
}

TEST(NodeLabels, InvalidInputThrowsOnEveryRank) {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::HEXAHEDRON);
    auto pmesh = Distribute(serial);
    mfem::Array<int> labels(pmesh->GetNE());
    labels = 1;
    if (pmesh->GetNE() > 0) {
        labels[0] = 0;
    }
    EXPECT_THROW(ms::NodeLabels(*pmesh, labels), std::invalid_argument);
    labels = 1;
    ms::NodeLabelOptions options;
    options.feature_angle_degrees = 90.0;
    EXPECT_THROW(ms::NodeLabels(*pmesh, labels, options), std::invalid_argument);
    mfem::Array<int> short_labels(0);
    EXPECT_THROW(ms::NodeLabels(*pmesh, short_labels), std::invalid_argument);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
