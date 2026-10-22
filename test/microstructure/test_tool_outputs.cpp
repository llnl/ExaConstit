/**
 * @file test_tool_outputs.cpp
 * @brief Tests of the generator tool's per-grain outputs:
 *   - ComputeGrainStatistics (labeling/grain_statistics.hpp): element counts, volumes, centroids,
 *     neighbor counts and shared face areas of grains, checked against hand-computed values on a
 *     hex box, against a brute-force serial loop on a tet mesh, with periodic face pairing, and
 *     for bitwise rank-count independence (METIS partition, a scattered partition with an empty
 *     rank, and a single-rank MPI_COMM_SELF reference computed in the same run);
 *   - CompareWithTargetVolumes: element-level seed volumes against size-control targets;
 *   - the new grain-table columns (neighbors and target volumes; Stage 1d phase, filler and voxel
 *     feature id) and the grain adjacency CSV writer (io/output_writer.hpp).
 *
 * The end-to-end runs of the tool through mechanics are CTest chains in test_tool_outputs.cmake
 * (Stages 1b/1c) and test_tool_stage1d.cmake (Stage 1d).
 * Registered at 1-4 MPI ranks.
 */
#include "microstructure/io/output_writer.hpp"
#include "microstructure/labeling/grain_statistics.hpp"
#include "microstructure/mesh/mesh_adjacency.hpp"
#include "microstructure/mesh/serial_order.hpp"
#include "microstructure/microstructure_types.hpp"
#include "utilities/auto_mesh.hpp"

#include "mfem.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace ms = exaconstit::microstructure;

namespace {

const fs::path data_dir = fs::path(MS_TOOL_OUTPUTS_DATA_DIR);

int Rank(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

/** A distributed mesh with its serial-order bookkeeping. */
struct Distributed {
    ms::SerialOrder order;
    std::unique_ptr<mfem::ParMesh> pmesh;
};

/** METIS partition of `serial` over `comm` (collective). */
Distributed Distribute(mfem::Mesh& serial, MPI_Comm comm) {
    Distributed d;
    d.order = ms::BuildSerialOrder(serial, comm);
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/**
 * Elements dealt round-robin to ranks 0 .. nranks-2, so the last rank (np >= 2) owns nothing:
 * many shared faces and an empty rank in every collective step. Collective.
 */
Distributed DistributeWithEmptyLastRank(mfem::Mesh& serial, MPI_Comm comm) {
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const int owners = (nranks > 1) ? nranks - 1 : 1;
    Distributed d;
    d.order.partitioning.SetSize(serial.GetNE());
    int num_local = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        d.order.partitioning[e] = e % owners;
        num_local += (e % owners == rank) ? 1 : 0;
    }
    d.order.local_to_serial.SetSize(num_local);
    int next = 0;
    for (int e = 0; e < serial.GetNE(); ++e) {
        if (d.order.partitioning[e] == rank) {
            d.order.local_to_serial[next++] = e;
        }
    }
    d.pmesh = ms::MakeParMesh(comm, serial, d.order);
    return d;
}

/** Average of the element's vertex coordinates (the same bits on every partition). */
std::array<double, 3> VertexAverage(const mfem::Mesh& mesh, int e) {
    mfem::Array<int> vertices;
    mesh.GetElementVertices(e, vertices);
    std::array<double, 3> c = {0.0, 0.0, 0.0};
    for (int k = 0; k < vertices.Size(); ++k) {
        const double* x = mesh.GetVertex(vertices[k]);
        for (std::size_t d = 0; d < 3; ++d) {
            c[d] += x[d];
        }
    }
    for (double& v : c) {
        v /= vertices.Size();
    }
    return c;
}

using LabelFunction = std::function<int(const std::array<double, 3>&)>;

/** Label of every local element from its vertex average. */
mfem::Array<int> LabelElements(const mfem::Mesh& mesh, const LabelFunction& label) {
    mfem::Array<int> labels(mesh.GetNE());
    for (int e = 0; e < mesh.GetNE(); ++e) {
        labels[e] = label(VertexAverage(mesh, e));
    }
    return labels;
}

/** Periodic pairing for the given axes on the box [lo, hi]. */
ms::PeriodicFaceParameters Pairing(const std::array<bool, 3>& periodic,
                                   const std::array<double, 3>& lo,
                                   const std::array<double, 3>& hi) {
    ms::PeriodicFaceParameters p;
    p.periodic = periodic;
    p.lo = lo;
    p.hi = hi;
    return p;
}

/**
 * Statistics of `serial` labeled by `label`, distributed over `comm` (METIS or the empty-rank
 * partition), with optional periodic pairing. Collective over comm.
 */
ms::GrainStatistics Statistics(mfem::Mesh& serial, MPI_Comm comm, const LabelFunction& label,
                               int num_grains, bool empty_last_rank = false,
                               const ms::PeriodicFaceParameters* periodic = nullptr) {
    Distributed d = empty_last_rank ? DistributeWithEmptyLastRank(serial, comm)
                                    : Distribute(serial, comm);
    std::unique_ptr<ms::MeshAdjacency> adjacency =
        periodic ? std::make_unique<ms::MeshAdjacency>(*d.pmesh, *periodic)
                 : std::make_unique<ms::MeshAdjacency>(*d.pmesh);
    return ms::ComputeGrainStatistics(*adjacency, LabelElements(*d.pmesh, label), num_grains);
}

/**
 * Every field of two statistics bitwise equal (doubles compared with ==). The pair table exists
 * only on the root rank of the computing communicator, so `with_pairs` says whether `a` holds it
 * (rank 0 of MPI_COMM_WORLD for a distributed run); elsewhere `a`'s pair table must be empty.
 */
void ExpectIdentical(const ms::GrainStatistics& a, const ms::GrainStatistics& b,
                     bool with_pairs = true) {
    EXPECT_EQ(a.num_grains, b.num_grains);
    EXPECT_EQ(a.num_elements, b.num_elements);
    EXPECT_EQ(a.volume, b.volume);
    EXPECT_EQ(a.centroid, b.centroid);
    EXPECT_EQ(a.num_neighbors, b.num_neighbors);
    EXPECT_EQ(a.total_volume, b.total_volume);
    if (with_pairs) {
        EXPECT_EQ(a.pair_a, b.pair_a);
        EXPECT_EQ(a.pair_b, b.pair_b);
        EXPECT_EQ(a.pair_area, b.pair_area);
        EXPECT_EQ(a.pair_faces, b.pair_faces);
    } else {
        EXPECT_TRUE(a.pair_a.empty() && a.pair_b.empty() && a.pair_area.empty() &&
                    a.pair_faces.empty());
    }
}

/** True on the rank that receives the pair table of a distributed (MPI_COMM_WORLD) run. */
bool WorldRoot() {
    return Rank() == 0;
}

/**
 * Scratch directory for the writer tests, unique per rank count: CTest runs this executable at
 * 1, 2, 3 and 4 ranks, possibly at the same time (ctest -j), so a shared directory name would let
 * one run delete the files another is still checking.
 */
fs::path ScratchDir(const std::string& base) {
    int nranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    return fs::current_path() / (base + "_np" + std::to_string(nranks));
}

/** Two grains on ExaConstit's 4x4x4 auto box: x < 0.5 is grain 1, x > 0.5 grain 2. */
int HalfLabel(const std::array<double, 3>& c) {
    return (c[0] < 0.5) ? 1 : 2;
}

/** Reads a whole text file. */
std::string ReadText(const fs::path& path) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

} // namespace

// Two halves of the unit cube (4x4x4 hexes, element size 1/4): each grain has 32 elements,
// volume 1/2 and centroid (0.25 or 0.75, 0.5, 0.5); the only shared surface is the plane x = 0.5,
// made of 16 element faces of area 1/16, so one pair (1, 2) with area 1 and 16 faces, and each
// grain has exactly 1 neighbor. The METIS partition, the empty-rank partition and the
// single-rank run must agree bit for bit.
TEST(GrainStatistics, HalvesOfAHexBox) {
    mfem::Mesh serial = exaconstit::MakeAutoHexMesh({4, 4, 4}, {1.0, 1.0, 1.0});
    // `alone` is computed by every rank on its own (MPI_COMM_SELF) from identical input, so the
    // ASSERTs below give the same verdict on every rank: no rank can return early alone and
    // leave the others waiting in the collective WORLD runs further down.
    const ms::GrainStatistics alone = Statistics(serial, MPI_COMM_SELF, HalfLabel, 2);
    ASSERT_EQ(alone.num_grains, 2);
    EXPECT_EQ(alone.num_elements, (std::vector<std::int64_t>{32, 32}));
    EXPECT_NEAR(alone.volume[0], 0.5, 1e-14);
    EXPECT_NEAR(alone.volume[1], 0.5, 1e-14);
    EXPECT_NEAR(alone.total_volume, 1.0, 1e-14);
    const double expected_centroid[6] = {0.25, 0.5, 0.5, 0.75, 0.5, 0.5};
    for (std::size_t k = 0; k < 6; ++k) {
        EXPECT_NEAR(alone.centroid[k], expected_centroid[k], 1e-12) << "component " << k;
    }
    EXPECT_EQ(alone.num_neighbors, (std::vector<int>{1, 1}));
    ASSERT_EQ(alone.pair_a.size(), 1u);
    EXPECT_EQ(alone.pair_a[0], 1);
    EXPECT_EQ(alone.pair_b[0], 2);
    EXPECT_EQ(alone.pair_faces[0], 16);
    EXPECT_NEAR(alone.pair_area[0], 1.0, 1e-13);

    for (const bool empty_last_rank : {false, true}) {
        SCOPED_TRACE(empty_last_rank ? "empty last rank" : "METIS partition");
        ExpectIdentical(Statistics(serial, MPI_COMM_WORLD, HalfLabel, 2, empty_last_rank), alone,
                        WorldRoot());
    }
}

// The same halves with periodic face pairing. Along x the planes x = 0 (grain 1) and x = 1
// (grain 2) are paired, so the grains also touch across the box face: 16 more faces, area 2 in
// total. Along y (or z) the paired faces join elements of the same grain, which adds nothing.
// A grain that is its own periodic neighbor is not a pair. Rank-count independent.
TEST(GrainStatistics, PeriodicPairsAddWrappedFaces) {
    mfem::Mesh serial = exaconstit::MakeAutoHexMesh({4, 4, 4}, {1.0, 1.0, 1.0});
    const std::array<double, 3> lo = {0.0, 0.0, 0.0};
    const std::array<double, 3> hi = {1.0, 1.0, 1.0};

    const ms::PeriodicFaceParameters x_periodic = Pairing({true, false, false}, lo, hi);
    // Single-rank (MPI_COMM_SELF) results from identical input: the ASSERTs agree on all ranks
    const ms::GrainStatistics alone_x =
        Statistics(serial, MPI_COMM_SELF, HalfLabel, 2, false, &x_periodic);
    ASSERT_EQ(alone_x.pair_a.size(), 1u);
    EXPECT_EQ(alone_x.pair_faces[0], 32);
    EXPECT_NEAR(alone_x.pair_area[0], 2.0, 1e-13);
    EXPECT_EQ(alone_x.num_neighbors, (std::vector<int>{1, 1}));

    const ms::PeriodicFaceParameters yz_periodic = Pairing({false, true, true}, lo, hi);
    const ms::GrainStatistics alone_yz =
        Statistics(serial, MPI_COMM_SELF, HalfLabel, 2, false, &yz_periodic);
    ASSERT_EQ(alone_yz.pair_a.size(), 1u);
    EXPECT_EQ(alone_yz.pair_faces[0], 16);
    EXPECT_NEAR(alone_yz.pair_area[0], 1.0, 1e-13);

    for (const bool empty_last_rank : {false, true}) {
        SCOPED_TRACE(empty_last_rank ? "empty last rank" : "METIS partition");
        ExpectIdentical(
            Statistics(serial, MPI_COMM_WORLD, HalfLabel, 2, empty_last_rank, &x_periodic),
            alone_x, WorldRoot());
    }
}

// cube_tet.mesh cut into 8 grains by three planes (offset from 0.5 so no vertex average lies on
// a plane). A brute-force loop on the serial mesh gives the reference: per grain the element
// count (exact), the volume (Mesh::GetElementVolume) and the volume-weighted centroid; per pair
// of different labels across an interior face the face count (exact) and the area (quadrature
// of the face transformation's area element, as MeshAdjacency does). The statistics must match
// to round-off, and be bitwise identical for every partition and rank count: the face areas
// come from canonically ordered vertices, so a face stored with another orientation on another
// rank still gives the same bits.
TEST(GrainStatistics, TetMeshMatchesBruteForce) {
    mfem::Mesh serial((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
    const LabelFunction octant = [](const std::array<double, 3>& c) {
        return 1 + (c[0] > 0.47 ? 1 : 0) + (c[1] > 0.53 ? 2 : 0) + (c[2] > 0.51 ? 4 : 0);
    };
    const int num_grains = 8;

    // Brute force on the serial mesh
    std::vector<std::int64_t> count(num_grains, 0);
    std::vector<double> volume(num_grains, 0.0);
    std::vector<double> moment(3 * num_grains, 0.0);
    const mfem::Array<int> labels = LabelElements(serial, octant);
    for (int e = 0; e < serial.GetNE(); ++e) {
        const auto g = static_cast<std::size_t>(labels[e] - 1);
        const double v = serial.GetElementVolume(e);
        // For a linear tet the volume centroid is the vertex average
        const std::array<double, 3> c = VertexAverage(serial, e);
        count[g] += 1;
        volume[g] += v;
        for (std::size_t d = 0; d < 3; ++d) {
            moment[3 * g + d] += v * c[d];
        }
    }
    std::map<std::pair<int, int>, std::pair<double, std::int64_t>> pairs;
    for (int f = 0; f < serial.GetNumFaces(); ++f) {
        int e1 = -1;
        int e2 = -1;
        serial.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 < 0 || labels[e1] == labels[e2]) {
            continue;
        }
        mfem::IsoparametricTransformation t;
        serial.GetFaceTransformation(f, &t);
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(serial.GetFaceGeometry(f), 3);
        double area = 0.0;
        for (int q = 0; q < rule.GetNPoints(); ++q) {
            t.SetIntPoint(&rule.IntPoint(q));
            area += rule.IntPoint(q).weight * t.Weight();
        }
        auto& entry = pairs[{std::min(labels[e1], labels[e2]), std::max(labels[e1], labels[e2])}];
        entry.first += area;
        entry.second += 1;
    }

    // The brute force and `alone` (MPI_COMM_SELF) are computed by every rank from identical
    // input, so the ASSERTs below give the same verdict on every rank
    const ms::GrainStatistics alone = Statistics(serial, MPI_COMM_SELF, octant, num_grains);
    EXPECT_EQ(alone.num_elements, count);
    for (std::size_t g = 0; g < static_cast<std::size_t>(num_grains); ++g) {
        ASSERT_GT(count[g], 0) << "every octant holds elements";
        EXPECT_NEAR(alone.volume[g], volume[g], 1e-13) << "grain " << g + 1;
        for (std::size_t d = 0; d < 3; ++d) {
            EXPECT_NEAR(alone.centroid[3 * g + d], moment[3 * g + d] / volume[g], 1e-12)
                << "grain " << g + 1 << " component " << d;
        }
    }
    ASSERT_EQ(alone.pair_a.size(), pairs.size());
    std::vector<int> neighbors(num_grains, 0);
    std::size_t k = 0;
    for (const auto& [key, value] : pairs) {
        EXPECT_EQ(alone.pair_a[k], key.first);
        EXPECT_EQ(alone.pair_b[k], key.second);
        EXPECT_EQ(alone.pair_faces[k], value.second);
        EXPECT_NEAR(alone.pair_area[k], value.first, 1e-13);
        neighbors[static_cast<std::size_t>(key.first - 1)]++;
        neighbors[static_cast<std::size_t>(key.second - 1)]++;
        ++k;
    }
    EXPECT_EQ(alone.num_neighbors, neighbors);

    for (const bool empty_last_rank : {false, true}) {
        SCOPED_TRACE(empty_last_rank ? "empty last rank" : "METIS partition");
        ExpectIdentical(Statistics(serial, MPI_COMM_WORLD, octant, num_grains, empty_last_rank),
                        alone, WorldRoot());
    }
}

/**
 * Area of face f of `mesh` as MFEM computes it from the stored face orientation (quadrature of
 * the face transformation's area element, as MeshAdjacency does for linear meshes).
 */
double StoredOrientationFaceArea(const mfem::Mesh& mesh, int f) {
    mfem::IsoparametricTransformation t;
    mesh.GetFaceTransformation(f, &t);
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(mesh.GetFaceGeometry(f), 3);
    double area = 0.0;
    for (int q = 0; q < rule.GetNPoints(); ++q) {
        t.SetIntPoint(&rule.IntPoint(q));
        area += rule.IntPoint(q).weight * t.Weight();
    }
    return area;
}

// The shared face areas must not depend on how a mesh stores its faces. Two copies of
// cube_tet.mesh are built with identical vertices (moved by a small smooth non-dyadic
// perturbation, so coordinates and areas are not exact binary fractions and rounding shows) and
// identical elements, but in copy B every tet's vertex list is rotated by one of the 12 even
// permutations (same orientation, so the mesh is unchanged geometrically). MFEM derives each
// face's stored vertex order from its first element's local vertex order, so the faces are
// stored differently in A and B. The test first checks that this is visible: some face area
// computed from the stored orientation differs in its bits between A and B. Then the grain
// statistics of the two copies (same labels) must have bitwise identical pair areas, which only
// holds because ComputeGrainStatistics orders the face corners canonically. (Mutation-checked:
// with the canonical ordering removed this test fails.) A third copy with order-1 Nodes must
// match too: order-1 Nodes are straight-sided and use the same canonical path. Each rank works alone (MPI_COMM_SELF)
// on identical input, so the ASSERTs give the same verdict on every rank.
TEST(GrainStatistics, FaceAreasIndependentOfStoredFaceOrientation) {
    mfem::Mesh original((data_dir / "cube_tet.mesh").string().c_str(), 1, 1, true);
    const int nv = original.GetNV();
    const int ne = original.GetNE();

    // Perturbed vertices (the same for both copies)
    std::vector<std::array<double, 3>> vertex(static_cast<std::size_t>(nv));
    for (int v = 0; v < nv; ++v) {
        const double* x = original.GetVertex(v);
        const double phase = 12.9898 * x[0] + 78.233 * x[1] + 37.719 * x[2];
        for (std::size_t d = 0; d < 3; ++d) {
            vertex[static_cast<std::size_t>(v)][d] =
                1.1 * x[d] + 0.003 * std::sin(phase + 1.7 * static_cast<double>(d));
        }
    }

    // The 12 even permutations of (0, 1, 2, 3): orientation-preserving vertex reorderings
    const int even[12][4] = {{0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 0, 3, 2},
                             {1, 2, 0, 3}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 1, 3, 0},
                             {2, 3, 0, 1}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 2, 1, 0}};
    auto build = [&](bool rotate) {
        mfem::Mesh mesh(3, nv, ne, 0, 3);
        for (const auto& x : vertex) {
            mesh.AddVertex(x[0], x[1], x[2]);
        }
        mfem::Array<int> tet;
        for (int e = 0; e < ne; ++e) {
            original.GetElementVertices(e, tet);
            const int* perm = even[rotate ? (e % 12) : 0];
            const int v[4] = {tet[perm[0]], tet[perm[1]], tet[perm[2]], tet[perm[3]]};
            mesh.AddTet(v, 1);
        }
        // refine = 0: keep the vertex order given (refine = 1 would reorder the tets)
        mesh.FinalizeTetMesh(1, 0, true);
        return mesh;
    };
    mfem::Mesh mesh_a = build(false);
    mfem::Mesh mesh_b = build(true);

    // The stored face orientations really differ: some face's area bits change
    auto face_areas = [](const mfem::Mesh& mesh) {
        std::map<std::array<int, 3>, double> areas;
        mfem::Array<int> fv;
        for (int f = 0; f < mesh.GetNumFaces(); ++f) {
            mesh.GetFaceVertices(f, fv);
            std::array<int, 3> key = {fv[0], fv[1], fv[2]};
            std::sort(key.begin(), key.end());
            areas[key] = StoredOrientationFaceArea(mesh, f);
        }
        return areas;
    };
    const auto areas_a = face_areas(mesh_a);
    const auto areas_b = face_areas(mesh_b);
    ASSERT_EQ(areas_a.size(), areas_b.size());
    int differing = 0;
    for (const auto& [key, area] : areas_a) {
        differing += (areas_b.at(key) != area) ? 1 : 0;
    }
    ASSERT_GT(differing, 0) << "the two copies must store some faces with another orientation";

    // Same labels (octants of the original element centers) on both copies
    const LabelFunction octant = [](const std::array<double, 3>& c) {
        return 1 + (c[0] > 0.47 ? 1 : 0) + (c[1] > 0.53 ? 2 : 0) + (c[2] > 0.51 ? 4 : 0);
    };
    const mfem::Array<int> labels = LabelElements(original, octant);
    auto statistics = [&](mfem::Mesh& serial) {
        Distributed d = Distribute(serial, MPI_COMM_SELF);
        ms::MeshAdjacency adjacency(*d.pmesh);
        mfem::Array<int> local(d.pmesh->GetNE());
        for (int e = 0; e < local.Size(); ++e) {
            local[e] = labels[d.order.local_to_serial[e]];
        }
        return ms::ComputeGrainStatistics(adjacency, local, 8);
    };
    const ms::GrainStatistics stats_a = statistics(mesh_a);
    const ms::GrainStatistics stats_b = statistics(mesh_b);
    ASSERT_FALSE(stats_a.pair_a.empty());
    EXPECT_EQ(stats_a.pair_a, stats_b.pair_a);
    EXPECT_EQ(stats_a.pair_b, stats_b.pair_b);
    EXPECT_EQ(stats_a.pair_faces, stats_b.pair_faces);
    EXPECT_EQ(stats_a.pair_area, stats_b.pair_area);
    EXPECT_EQ(stats_a.num_neighbors, stats_b.num_neighbors);

    // Order-1 Nodes (as for a mesh read with a nodes section) describe the same straight-sided
    // geometry, so they must take the same canonical path and give the same bits
    mfem::Mesh mesh_c = build(true);
    mesh_c.SetCurvature(1);
    ASSERT_NE(mesh_c.GetNodes(), nullptr);
    const ms::GrainStatistics stats_c = statistics(mesh_c);
    EXPECT_EQ(stats_a.pair_area, stats_c.pair_area);
    EXPECT_EQ(stats_a.pair_faces, stats_c.pair_faces);
}

// Bad input throws on every rank together: a grain id outside 1..num_grains on the last rank
// only (CellMeasureAccumulator reduces the check), num_grains < 1, and a root that is not a
// rank. A valid call afterwards proves that no rank was left behind in a collective step.
TEST(GrainStatistics, BadLabelsThrowOnAllRanks) {
    mfem::Mesh serial = exaconstit::MakeAutoHexMesh({4, 4, 4}, {1.0, 1.0, 1.0});
    Distributed d = Distribute(serial, MPI_COMM_WORLD);
    ms::MeshAdjacency adjacency(*d.pmesh);
    int nranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    mfem::Array<int> labels = LabelElements(*d.pmesh, HalfLabel);
    if (Rank() == nranks - 1 && labels.Size() > 0) {
        labels[0] = 3;
    }
    EXPECT_THROW(ms::ComputeGrainStatistics(adjacency, labels, 2), std::invalid_argument);
    EXPECT_THROW(ms::ComputeGrainStatistics(adjacency, labels, 0), std::invalid_argument);
    EXPECT_THROW(ms::ComputeGrainStatistics(adjacency, labels, 2, nranks), std::invalid_argument);

    // The ranks are still in step after the throws: a valid collective call completes on every
    // rank and gives the two halves of HalvesOfAHexBox
    const ms::GrainStatistics ok =
        ms::ComputeGrainStatistics(adjacency, LabelElements(*d.pmesh, HalfLabel), 2);
    EXPECT_EQ(ok.num_elements, (std::vector<std::int64_t>{32, 32}));
    EXPECT_EQ(ok.num_neighbors, (std::vector<int>{1, 1}));
}

// Element-level seed volumes against targets. Grains 1 and 2 descend from seed 0 (a split
// grain), grain 3 from seed 2; seed 1 lost all its elements. Seed volumes: 1 + 2 = 3, 0, 3 against
// targets 3, 1, 2 give relative errors 0, 1 (empty seed) and 0.5: max 1 at seed 1, mean 0.5.
TEST(TargetVolumes, ElementLevelErrors) {
    const ms::TargetVolumeComparison cmp =
        ms::CompareWithTargetVolumes({1.0, 2.0, 3.0}, {0, 0, 2}, {3.0, 1.0, 2.0});
    EXPECT_EQ(cmp.num_seeds, 3);
    EXPECT_EQ(cmp.seed_volume, (std::vector<double>{3.0, 0.0, 3.0}));
    EXPECT_EQ(cmp.num_empty_seeds, 1);
    EXPECT_DOUBLE_EQ(cmp.max_relative_error, 1.0);
    EXPECT_EQ(cmp.worst_seed, 1);
    EXPECT_DOUBLE_EQ(cmp.mean_relative_error, 0.5);

    // Perfect volumes: zero error
    const ms::TargetVolumeComparison exact =
        ms::CompareWithTargetVolumes({0.25, 0.75}, {1, 0}, {0.75, 0.25});
    EXPECT_EQ(exact.max_relative_error, 0.0);
    EXPECT_EQ(exact.num_empty_seeds, 0);

    // Inconsistent input
    EXPECT_THROW(ms::CompareWithTargetVolumes({1.0}, {0, 1}, {1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(ms::CompareWithTargetVolumes({1.0}, {2}, {1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(ms::CompareWithTargetVolumes({1.0}, {0}, {0.0}), std::invalid_argument);
}

// The writer's new outputs: the grain table appends num_neighbors and target_volume after the
// existing columns; the adjacency CSV has one row per pair with 17 significant digits; wrong
// column lengths and unordered pairs are rejected before anything is written. Rank 0 only.
TEST(OutputWriter, NeighborColumnsAndAdjacencyCsv) {
    if (Rank() != 0) {
        return;
    }
    const fs::path dir = ScratchDir("test_tool_outputs_scratch");
    fs::create_directories(dir);

    ms::GrainTable table;
    table.num_grains = 2;
    table.quat = {1, 0, 0, 0, 0, 1, 0, 0};
    table.num_elements = {3, 5};
    table.volume = {0.375, 0.625};
    table.num_neighbors = {1, 1};
    table.target_volume = {0.5, 0.5};
    ms::WriteGrainTable(dir / "grains.csv", table);
    EXPECT_EQ(ReadText(dir / "grains.csv"),
              "grain_id,qw,qx,qy,qz,num_elements,volume,num_neighbors,target_volume\n"
              "1,1,0,0,0,3,0.375,1,0.5\n"
              "2,0,1,0,0,5,0.625,1,0.5\n");
    table.num_neighbors = {1};
    EXPECT_THROW(ms::WriteGrainTable(dir / "bad.csv", table), std::invalid_argument);

    ms::GrainAdjacencyTable adjacency;
    adjacency.grain_a = {1, 1};
    adjacency.grain_b = {2, 3};
    adjacency.shared_area = {0.1, 2.5};
    adjacency.num_faces = {4, 40};
    ms::WriteGrainAdjacency(dir / "adjacency.csv", adjacency);
    EXPECT_EQ(ReadText(dir / "adjacency.csv"),
              "grain_a,grain_b,shared_area,num_faces\n"
              "1,2,0.10000000000000001,4\n"
              "1,3,2.5,40\n");

    ms::GrainAdjacencyTable unordered = adjacency;
    unordered.grain_b[1] = 1;
    EXPECT_THROW(ms::WriteGrainAdjacency(dir / "bad.csv", unordered), std::invalid_argument);
    ms::GrainAdjacencyTable short_column = adjacency;
    short_column.num_faces.pop_back();
    EXPECT_THROW(ms::WriteGrainAdjacency(dir / "bad.csv", short_column), std::invalid_argument);
    EXPECT_FALSE(fs::exists(dir / "bad.csv"));
    fs::remove_all(dir);
}

// The Stage 1d grain-table columns follow the optional-column rule: MakeGrainTable copies the
// phase column only when some grain has a phase (-1 = none, as for a passthrough grain) and the
// filler column only when some grain is a filler grain; the caller adds feature ids for voxel
// maps. The new columns come after the older ones (phase, filler, feature_id), so existing column
// positions do not move. Grain 3 plays a passthrough grain (phase -1, feature id -1). Rank 0 only.
TEST(OutputWriter, PhaseFillerAndFeatureColumns) {
    if (Rank() != 0) {
        return;
    }
    const fs::path dir = ScratchDir("test_tool_outputs_stage1d_scratch");
    fs::create_directories(dir);

    ms::MicrostructureResult result;
    result.num_grains = 3;
    result.grain_quat = {1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0};
    result.grain_region = {1, 2, 3};
    result.grain_source_id = {5, 7, -2};
    // No phase anywhere and no filler grain: neither column is copied
    result.grain_phase = {-1, -1, -1};
    result.grain_filler = {0, 0, 0};
    ms::GrainTable plain = ms::MakeGrainTable(result);
    EXPECT_TRUE(plain.phase.empty());
    EXPECT_TRUE(plain.filler.empty());
    ms::WriteGrainTable(dir / "plain.csv", plain);
    EXPECT_EQ(ReadText(dir / "plain.csv"),
              "grain_id,qw,qx,qy,qz,region,source_id\n"
              "1,1,0,0,0,1,5\n"
              "2,0,1,0,0,2,7\n"
              "3,1,0,0,0,3,-2\n");

    // Phases on the first two grains and one filler grain: both columns, plus feature ids
    result.grain_phase = {0, 4, -1};
    result.grain_filler = {0, 1, 0};
    ms::GrainTable table = ms::MakeGrainTable(result);
    EXPECT_EQ(table.phase, (std::vector<int>{0, 4, -1}));
    EXPECT_EQ(table.filler, (std::vector<int>{0, 1, 0}));
    table.num_elements = {2, 3, 4};
    table.feature_id = {21, 3, -1};
    ms::WriteGrainTable(dir / "stage1d.csv", table);
    EXPECT_EQ(ReadText(dir / "stage1d.csv"),
              "grain_id,qw,qx,qy,qz,region,source_id,num_elements,phase,filler,feature_id\n"
              "1,1,0,0,0,1,5,2,0,0,21\n"
              "2,0,1,0,0,2,7,3,4,1,3\n"
              "3,1,0,0,0,3,-2,4,-1,0,-1\n");

    // A wrong length of a new column is rejected before anything is written
    table.feature_id = {21, 3};
    EXPECT_THROW(ms::WriteGrainTable(dir / "bad.csv", table), std::invalid_argument);
    table.feature_id.clear();
    table.phase = {1};
    EXPECT_THROW(ms::WriteGrainTable(dir / "bad.csv", table), std::invalid_argument);
    EXPECT_FALSE(fs::exists(dir / "bad.csv"));
    fs::remove_all(dir);
}

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
